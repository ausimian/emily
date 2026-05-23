defmodule Emily.Memory do
  @moduledoc """
  Public MLX allocator controls.

  Emily tensors hold MLX-managed buffers outside the BEAM heap. This
  module exposes the small set of allocator hooks that are useful when
  running long-lived serving or training workloads:

    * `stats/0` samples active, peak, and cached MLX memory.
    * `reset_peak/0` resets the peak-memory high-water mark.
    * `clear_cache/0` asks MLX to release cached reusable buffers.

  `stats/0` also emits `[:emily, :memory, :stats]`, so callers can use
  it directly in a periodic telemetry loop.

  ## Examples

      iex> stats = Emily.Memory.stats()
      iex> Map.keys(stats) |> Enum.sort()
      [:active, :cache, :peak]

      iex> Emily.Memory.reset_peak()
      :ok

      iex> Emily.Memory.clear_cache()
      :ok

  """

  alias Emily.Native

  @typedoc """
  MLX allocator measurements in bytes.

    * `:active` — bytes currently held by live MLX arrays.
    * `:peak` — high-water mark since the last `reset_peak/0`.
    * `:cache` — bytes retained by MLX for reuse.
  """
  @type stats :: %{
          active: non_neg_integer(),
          peak: non_neg_integer(),
          cache: non_neg_integer()
        }

  @doc """
  Sample MLX allocator memory and emit `[:emily, :memory, :stats]`.

  This is a discrete telemetry event, not a span. It is intended for
  poll-driven monitoring in long-running processes.
  """
  @spec stats() :: stats()
  def stats do
    measurements = %{
      active: Native.get_active_memory(),
      peak: Native.get_peak_memory(),
      cache: Native.get_cache_memory()
    }

    :telemetry.execute([:emily, :memory, :stats], measurements, %{})
    measurements
  end

  @doc """
  Reset the allocator peak-memory high-water mark.
  """
  @spec reset_peak() :: :ok
  def reset_peak, do: Native.reset_peak_memory()

  @doc """
  Ask MLX to release cached reusable buffers.

  This does not free live tensors. Tensors and resource binaries still
  retain their underlying MLX buffers until the owning BEAM references
  are garbage collected.
  """
  @spec clear_cache() :: :ok
  def clear_cache, do: Native.clear_cache()
end
