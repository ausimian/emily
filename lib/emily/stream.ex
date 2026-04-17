defmodule Emily.Stream do
  @moduledoc """
  Per-process MLX stream management for concurrent inference.

  MLX dispatches GPU work through Metal command queues. By default all
  operations share a single command queue (the default worker thread).
  `Emily.Stream` lets each BEAM process use its own worker thread —
  its own Metal command queue — so multiple processes can run inference
  concurrently on the same model without crashing.

  ## Usage

      stream = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(stream, fn ->
        # All Emily ops in this block dispatch on `stream`
        model.(input)
      end)

  ## How it works

  `with_stream/2` stores the worker reference in the process dictionary.
  `Emily.Backend` reads it via `Process.get(:emily_worker)` and passes
  it as an explicit argument to every NIF call. Each NIF dispatches
  work to the worker's dedicated OS thread where the MLX stream lives.

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

  @enforce_keys [:worker]
  defstruct [:worker]

  @type t :: %__MODULE__{worker: reference()}

  @doc """
  Create a new stream (Metal command queue) on the given device.

  Each stream is backed by a dedicated OS thread that owns the MLX
  stream and its Metal command encoder. The worker thread is cleaned
  up when the resource is garbage collected.
  """
  @spec new(:gpu | :cpu) :: t()
  def new(_device \\ :gpu) do
    worker = Emily.Native.create_worker()
    %__MODULE__{worker: worker}
  end

  @doc """
  Execute `fun` with the given stream as the default for MLX ops.

  Stores the worker reference in the process dictionary so that
  `Emily.Backend` passes it to every NIF call. The previous worker
  (if any) is restored in an `after` block, so nesting is safe.
  """
  @spec with_stream(t(), (-> result)) :: result when result: var
  def with_stream(%__MODULE__{worker: w}, fun) when is_function(fun, 0) do
    prev = Process.get(:emily_worker)
    Process.put(:emily_worker, w)

    try do
      fun.()
    after
      case prev do
        nil -> Process.delete(:emily_worker)
        prev_w -> Process.put(:emily_worker, prev_w)
      end
    end
  end
end
