defmodule Emily.Stream do
  @moduledoc """
  Per-process MLX stream management for concurrent inference.

  MLX dispatches GPU work through Metal command queues. By default
  every op shares a single command queue (the default worker thread).
  `Emily.Stream` lets each BEAM process use its own worker thread —
  its own Metal command queue — so multiple processes can run
  inference concurrently on a shared model.

  ## Public API

    * `new/1` — create a stream on `:gpu` or `:cpu`. Each stream
      allocates a dedicated OS thread that owns the MLX stream object;
      the thread is joined when the stream reference is garbage
      collected.
    * `with_stream/2` — install a stream for the current process for
      the duration of a function call, then restore the previous
      stream (or the default) on exit. Nesting is safe.

  ## How it works

  `with_stream/2` stores the worker reference in the process
  dictionary under `:emily_worker`. `Emily.Backend` reads it and
  passes it to every NIF call. Each NIF dispatches its work to the
  worker's dedicated OS thread where the MLX stream lives. Tensors
  allocated by one stream can be read by another (MLX arrays are
  refcounted and thread-safe for reads), but lazy tensors must be
  evaluated on the stream that created them.

  ## Concurrent serving patterns

  **Stream-per-process** (shared model, per-process queues):

      stream = Emily.Stream.new(:gpu)
      Emily.Stream.with_stream(stream, fn ->
        Nx.Serving.batched_run(my_serving, input)
      end)

  Each serving worker allocates its own stream once at init. Weights
  are shared — no duplication.

  **Pooled servings** (K instances behind a pool):

  Start K `Nx.Serving` instances behind poolboy / Registry / etc.
  Each loads its own weights and runs on the default stream. No
  `Emily.Stream` needed. Trade-off: each pool member holds its own
  weight copy, so memory scales with K.

  For small models the pool approach is simpler. For large models
  (Qwen3-7B+) where duplicating weights is impractical, use
  stream-per-process.

  ## Examples

      iex> stream = Emily.Stream.new(:gpu)
      iex> Emily.Stream.with_stream(stream, fn -> 42 end)
      42

  """

  @enforce_keys [:worker]
  defstruct [:worker]

  @type t :: %__MODULE__{worker: reference()}

  @doc """
  Create a new stream (Metal command queue) on the given device.

  Each stream is backed by a dedicated OS thread that owns the MLX
  stream and its Metal command encoder. The worker thread is cleaned
  up when the resource is garbage collected.

  ## Examples

      iex> stream = Emily.Stream.new(:gpu)
      iex> match?(%Emily.Stream{}, stream)
      true

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

  ## Examples

      iex> stream = Emily.Stream.new(:gpu)
      iex> Emily.Stream.with_stream(stream, fn ->
      ...>   Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)
      ...>   |> Nx.sum()
      ...>   |> Nx.to_number()
      ...> end)
      6.0

  """
  @spec with_stream(t(), (-> result)) :: result when result: var
  def with_stream(%__MODULE__{worker: w}, fun) when is_function(fun, 0) do
    prev = Process.put(:emily_worker, w)

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
