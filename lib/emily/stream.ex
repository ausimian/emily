defmodule Emily.Stream do
  @moduledoc """
  Per-process MLX stream management for concurrent inference.

  MLX dispatches GPU work through Metal command queues. By default all
  operations share a single command queue (the default stream), which
  is not safe for concurrent kernel dispatch from multiple OS threads.
  `Emily.Stream` lets each BEAM process use its own stream — its own
  Metal command queue — so multiple processes can run inference
  concurrently on the same model without crashing.

  ## Usage

      stream = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(stream, fn ->
        # All Emily ops in this block dispatch on `stream`
        model.(input)
      end)

  ## How it works

  `with_stream/2` stores the stream index in the process dictionary.
  `Emily.Backend` reads it via `Process.get(:emily_stream, -1)` and
  passes it as an explicit argument to every NIF call. Each NIF
  resolves the index to an `mx::Stream` (or falls back to the
  thread-local default when the index is -1). This avoids the
  thread-local race that would occur if we relied solely on
  `set_default_stream` — BEAM processes can migrate between OS
  scheduler threads between NIF calls.

  ## Concurrent serving patterns

  **Stream-per-process** (shared model, per-process queues):

      # Each serving process wraps its work in with_stream.
      # Weights are shared — no duplication.
      stream = Emily.Stream.new(:gpu)
      Emily.Stream.with_stream(stream, fn ->
        Nx.Serving.batched_run(my_serving, input)
      end)

  **Pooled servings** (K instances behind a pool):

      # Each pool member loads its own weights and runs on the
      # default stream. No Emily.Stream needed — just start K
      # Nx.Serving instances behind poolboy / Registry / etc.
      # Trade-off: each instance holds its own weight copy.

  For small models the pool approach is simpler. For large models
  (Qwen3-7B+) where duplicating weights is impractical, use
  stream-per-process.
  """

  @enforce_keys [:index, :device]
  defstruct [:index, :device]

  @type t :: %__MODULE__{index: integer(), device: :gpu | :cpu}

  @doc """
  Create a new stream (Metal command queue) on the given device.

  Streams are never freed by MLX — create them at init time (one per
  concurrent serving process), not per-request.
  """
  @spec new(:gpu | :cpu) :: t()
  def new(device \\ :gpu) when device in [:gpu, :cpu] do
    index = Emily.Native.new_stream(device)
    %__MODULE__{index: index, device: device}
  end

  @doc """
  Execute `fun` with the given stream as the default for MLX ops.

  Stores the stream index in the process dictionary so that
  `Emily.Backend` passes it to every NIF call. Also sets the
  thread-local default stream as a belt-and-suspenders measure for
  code that calls `Emily.Native` directly.

  The previous stream (if any) is restored in an `after` block, so
  nesting is safe.
  """
  @spec with_stream(t(), (-> result)) :: result when result: var
  def with_stream(%__MODULE__{index: index}, fun) when is_function(fun, 0) do
    prev = Process.get(:emily_stream)
    Process.put(:emily_stream, index)
    Emily.Native.set_default_stream(index)

    try do
      fun.()
    after
      case prev do
        nil -> Process.delete(:emily_stream)
        idx -> Process.put(:emily_stream, idx)
      end

      if prev, do: Emily.Native.set_default_stream(prev)
    end
  end

  @doc "Block until all operations on the stream have completed."
  @spec synchronize(t()) :: :ok
  def synchronize(%__MODULE__{index: index}) do
    Emily.Native.synchronize_stream(index)
  end
end
