defmodule Emily.Native.Async do
  @moduledoc false
  # Helper for awaiting async NIF replies.
  #
  # Async NIFs return a ref synchronously and dispatch the actual
  # work onto a worker thread that posts
  # `{ref, {:ok, result}}` or `{ref, {:error, reason}}` back to the
  # caller PID via `enif_send`. `call/1` awaits that message.
  #
  # Error reasons mirror `fine::nif_impl`'s sync catch ladder:
  #
  #   {:argument, binary} -> ArgumentError
  #   {:runtime, binary}  -> RuntimeError
  #   :unknown            -> RuntimeError
  #
  # See `c_src/emily/async.hpp` and
  # `docs/planning/async-worker-exploration.md`.

  @doc """
  Await the reply posted by the worker thread for `ref`.

  Blocks the calling process on `receive/1`, not any BEAM scheduler
  — the scheduler can run other work while the worker executes the
  op.
  """
  @spec call(reference()) :: term()
  def call(ref) do
    receive do
      {^ref, {:ok, result}} ->
        result

      {^ref, {:error, {:argument, message}}} ->
        raise ArgumentError, message

      {^ref, {:error, {:runtime, message}}} ->
        raise RuntimeError, message

      {^ref, {:error, :unknown}} ->
        raise RuntimeError, "unknown exception thrown within NIF"
    end
  end
end
