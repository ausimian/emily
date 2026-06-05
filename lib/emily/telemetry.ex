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

  Per-fallback behaviour is configurable via `:fallback`:

      config :emily, fallback: :silent | :warn | :raise

    * `:silent` (default) — span events still fire, no log, no raise.
      Library consumers and CI logs stay quiet.
    * `:warn` — one-shot `Logger.warning` per `{op, input_shapes}`
      pair. A Bumblebee user sees
      `"indexed_put on shape [...] fell back to Nx.BinaryBackend"`
      once per shape, not every forward pass. Typically set in
      `config/dev.exs` while chasing performance regressions.
    * `:raise` — raises a `RuntimeError` carrying the op, input
      shapes, and input dtypes. Use in CI to fail builds when a hot
      path unexpectedly routes through `Nx.BinaryBackend`.

  In `:raise` mode the `:start`/`:stop` span events do **not** fire
  because the raise happens on entry; `:silent` and `:warn` preserve
  the full span.

  The legacy `:warn_on_fallback` boolean is still honoured when
  `:fallback` is unset (`true` → `:warn`, `false` → `:silent`).
  Prefer `:fallback` in new code; if both are set, `:fallback` wins.

  ### Native-compiler fallback

  `[:emily, :compiler, :fallback]` — a discrete event (not a span) that
  fires when a `native: true` defn can't be lowered by the Expr compiler
  and routes through `Nx.Defn.Evaluator` instead (see
  `Emily.Compiler`'s `:native_fallback` option). Measurements:
  `:count` (always `1`). Metadata: `:key` (the JIT key) and `:reason`
  (the lowering error message, which names the unsupported op or
  construct). A one-shot `Logger.warning` per distinct `:reason` is also
  emitted — set `config :emily, :native_fallback, :raise` to fail
  instead of falling back.

  ### Memory stats (poll-driven)

  `[:emily, :memory, :stats]` — discrete event, not a span. Call
  `Emily.Memory.stats/0` to sample; measurements:

    * `:active` — bytes currently held by MLX
    * `:peak` — high-water mark since last `Emily.Memory.reset_peak/0`
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

  @dedup_table :emily_fallback_dedup

  @doc """
  Sample the MLX allocator and emit `[:emily, :memory, :stats]`.

  Prefer `Emily.Memory.stats/0` in new code. This function remains as
  the telemetry-oriented entry point for existing callers.

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
    Emily.Memory.stats()
  end

  @doc false
  # Called from `Emily.Backend.via_binary/4` and the tuple variant
  # before the telemetry span. Dispatches on the configured
  # `:fallback` mode:
  #
  #   * `:silent` — no-op
  #   * `:warn`   — one-shot Logger.warning per `{op, shapes}` pair,
  #                 deduped via ETS
  #   * `:raise`  — raises RuntimeError with op, shapes, dtypes
  @spec handle_fallback(atom(), [tuple()], [Nx.Type.t()]) :: :ok
  def handle_fallback(op, input_shapes, input_dtypes) do
    case fallback_mode() do
      :silent -> :ok
      :warn -> warn_fallback(op, input_shapes)
      :raise -> raise_fallback(op, input_shapes, input_dtypes)
    end
  end

  @doc false
  @spec fallback_mode() :: :silent | :warn | :raise
  def fallback_mode do
    case Application.get_env(:emily, :fallback) do
      mode when mode in [:silent, :warn, :raise] ->
        mode

      nil ->
        if Application.get_env(:emily, :warn_on_fallback, false),
          do: :warn,
          else: :silent

      other ->
        raise ArgumentError,
              "invalid :emily, :fallback config #{inspect(other)}; " <>
                "expected one of :silent | :warn | :raise"
    end
  end

  defp warn_fallback(op, input_shapes) do
    warn_once(
      {op, input_shapes},
      "Emily: #{op} on shapes #{inspect(input_shapes)} fell back to " <>
        "Nx.BinaryBackend; this path is ~100× slower than native MLX."
    )
  end

  # One-shot `Logger.warning` per distinct `key`, deduped via the shared
  # ETS table so a hot path that repeatedly falls back logs once, not per
  # call. Keys are namespaced by caller (`{op, shapes}` for the backend
  # fallback, `{:compiler, reason}` for the native-compiler fallback) so
  # the two never collide.
  defp warn_once(key, message) do
    if :ets.insert_new(@dedup_table, {key}) do
      Logger.warning(message)
    end

    :ok
  end

  defp raise_fallback(op, input_shapes, input_dtypes) do
    raise "Emily: #{op} fell back to Nx.BinaryBackend " <>
            "(shapes=#{inspect(input_shapes)}, dtypes=#{inspect(input_dtypes)}). " <>
            "Set `config :emily, fallback: :warn` to log instead, or `:silent` to ignore."
  end

  @doc false
  # Called from `Emily.Compiler` when a `native: true` defn cannot be
  # lowered by the Expr compiler and routes through `Nx.Defn.Evaluator`
  # instead. Fires the discrete `[:emily, :compiler, :fallback]` event
  # (always) and a one-shot `Logger.warning` per distinct reason, deduped
  # via the shared ETS table. `reason` is the lowering error message — it
  # names the unsupported op/construct.
  @spec compiler_fallback(term(), Exception.t()) :: :ok
  def compiler_fallback(key, exception) do
    reason = Exception.message(exception)

    :telemetry.execute(
      [:emily, :compiler, :fallback],
      %{count: 1},
      %{key: key, reason: reason}
    )

    warn_once(
      {:compiler, reason},
      "Emily: native compilation fell back to Nx.Defn.Evaluator — #{reason} " <>
        "The defn ran op-by-op via the evaluator. Set " <>
        "`config :emily, native_fallback: :raise` (or pass " <>
        "`native_fallback: :raise`) to fail instead."
    )
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
