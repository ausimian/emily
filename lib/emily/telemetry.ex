defmodule Emily.Telemetry do
  @moduledoc """
  `:telemetry` events emitted by Emily.

  All span-style events use `:telemetry.span/3` semantics, so attaching
  to `*:start`, `*:stop`, and `*:exception` is sufficient for
  histograms and error tracking.

  ## Events

  ### Evaluation boundaries

  `[:emily, :eval, :start | :stop | :exception]` — `Emily.eval/1`.
  The `:stop` event carries `:duration` (monotonic native units).

  `[:emily, :to_binary, :start | :stop | :exception]` — fires for both
  `Emily.to_binary/1` and the `Nx.to_binary/1` path on `Emily.Backend`.
  Metadata: `:byte_size`, `:shape`, `:dtype`.

  ### Fallback entry

  `[:emily, :fallback, :start | :stop | :exception]` — fires whenever
  an op routes through `Nx.BinaryBackend` because the MLX path is not
  wired. Metadata: `:op`, `:input_shapes`, `:input_dtypes`.

  A one-shot `Logger.warning` per `{op, input_shapes}` pair is
  **opt-in**: the span event fires on every fallback, but the log
  line is off by default so library consumers don't get unsolicited
  warnings. Turn it on — typically in `config/dev.exs` — when
  chasing performance regressions:

      config :emily, :warn_on_fallback, true

  With it on, a Bumblebee user sees
  `"indexed_put on shape [...] fell back to Nx.BinaryBackend"` once
  per shape, not every forward pass.

  ### Memory stats (poll-driven)

  `[:emily, :memory, :stats]` — discrete event, not a span. Call
  `Emily.Telemetry.memory_stats/0` to sample; measurements:

    * `:active` — bytes currently held by MLX
    * `:peak` — high-water mark since last `Native.reset_peak_memory/0`
    * `:cache` — bytes cached for reuse

  Wire this into a periodic task (e.g. `Process.send_after/3` loop) to
  graph memory drift in a long-running serving.

  ## Attaching a handler

      :telemetry.attach(
        "emily-fallback-log",
        [:emily, :fallback, :stop],
        fn _event, measurements, metadata, _config ->
          IO.inspect({metadata.op, measurements.duration})
        end,
        nil
      )
  """

  require Logger

  alias Emily.Native

  @dedup_table :emily_fallback_dedup

  @doc """
  Sample the MLX allocator and emit `[:emily, :memory, :stats]`.

  Returns the measurements map so callers can also log or plot
  inline.

  ## Examples

      iex> stats = Emily.Telemetry.memory_stats()
      iex> Map.keys(stats) |> Enum.sort()
      [:active, :cache, :peak]

  """
  @spec memory_stats() :: %{
          active: non_neg_integer(),
          peak: non_neg_integer(),
          cache: non_neg_integer()
        }
  def memory_stats do
    measurements = %{
      active: Native.get_active_memory(),
      peak: Native.get_peak_memory(),
      cache: Native.get_cache_memory()
    }

    :telemetry.execute([:emily, :memory, :stats], measurements, %{})
    measurements
  end

  @doc false
  # Called from `Emily.Backend.via_binary/4`. Idempotent via an ETS
  # set — first hit per `{op, shapes}` pair logs a warning; subsequent
  # hits no-op.
  @spec maybe_warn_fallback(atom(), [tuple()]) :: :ok
  def maybe_warn_fallback(op, input_shapes) do
    if Application.get_env(:emily, :warn_on_fallback, false) do
      key = {op, input_shapes}

      if :ets.insert_new(@dedup_table, {key}) do
        Logger.warning(
          "Emily: #{op} on shapes #{inspect(input_shapes)} fell back to " <>
            "Nx.BinaryBackend; this path is ~100× slower than native MLX."
        )
      end
    end

    :ok
  end

  @doc false
  # Called exactly once from `Emily.Application.start/2` before the
  # supervisor starts. Safe to call again; an existing table is
  # tolerated.
  @spec init_dedup_table() :: :ok
  def init_dedup_table do
    case :ets.whereis(@dedup_table) do
      :undefined ->
        _ =
          :ets.new(@dedup_table, [
            :set,
            :public,
            :named_table,
            read_concurrency: true,
            write_concurrency: true
          ])

        :ok

      _ref ->
        :ok
    end
  end

  @doc false
  # Exposed for tests that need to reset dedup state between cases.
  @spec reset_dedup() :: :ok
  def reset_dedup do
    case :ets.whereis(@dedup_table) do
      :undefined -> :ok
      _ -> :ets.delete_all_objects(@dedup_table)
    end

    :ok
  end
end
