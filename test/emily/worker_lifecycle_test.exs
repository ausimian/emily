defmodule Emily.WorkerLifecycleTest do
  @moduledoc """
  Tests for the worker queue/teardown behaviour (#121, #122): bounded
  back-pressure, drop-with-`:stopped` cancellation on stop, the explicit
  `Emily.Stream.close/1`, and that none of it leaves an awaiting process
  hanging.
  """

  # async: false — these tests measure global allocator state
  # (clear_cache / get_active_memory) and rely on a worker being busy
  # without competing GPU load, so they must not run concurrently with
  # other tests.
  use ExUnit.Case, async: false

  alias Emily.Native

  # A lazy tensor whose `eval` runs a chain of 1024×1024 matmuls — slow
  # enough (several ms) that the worker is still busy on the first eval
  # while we pile work up behind it.
  defp slow_lazy(w) do
    a = Native.random_normal(w, [1024, 1024], {:f, 32}, 0.0, 1.0, nil)
    Enum.reduce(1..4, a, fn _, acc -> Native.matmul(w, acc, a) end)
  end

  defp await_tag(ref, timeout \\ 5000) do
    receive do
      {^ref, {:ok, _}} -> :ok
      {^ref, {:error, :stopped}} -> :stopped
      {^ref, {:error, other}} -> {:error, other}
    after
      timeout -> :timeout
    end
  end

  test "create_worker rejects a non-positive queue limit" do
    assert_raise ArgumentError, fn -> Native.create_worker(0) end
    assert_raise ArgumentError, fn -> Native.create_worker(-1) end
  end

  test "worker_queue_depth of an idle worker is 0" do
    w = Native.create_worker(8192)
    assert Native.worker_queue_depth(w) == 0
    Native.stop_worker(w)
  end

  test "stop_worker is idempotent and rejects further work" do
    w = Native.create_worker(8192)
    assert :ok = Native.stop_worker(w)
    # Idempotent.
    assert :ok = Native.stop_worker(w)

    t = Native.from_binary(<<1.0::float-32-native>>, [1], {:f, 32})

    assert_raise RuntimeError, ~r/stopped/, fn -> Native.eval(w, t) end
  end

  test "stopping a worker cancels queued ops with :stopped and never hangs" do
    w = Native.create_worker(8192)
    big = slow_lazy(w)

    # First eval starts running (slow); the rest queue up behind it.
    ref1 = Native.eval_nif(w, big)
    refs = for _ <- 1..20, do: Native.eval_nif(w, big)

    Native.stop_worker(w)

    results = for ref <- [ref1 | refs], do: await_tag(ref)

    # Nothing is left hanging, and the cancelled backlog reports :stopped.
    refute :timeout in results
    assert :stopped in results
  end

  test "run_async applies back-pressure when the queue is full" do
    # Tiny cap so a burst of enqueues outruns the (slow) worker.
    w = Native.create_worker(4)
    big = slow_lazy(w)

    rejected =
      Enum.reduce(1..500, 0, fn _, acc ->
        try do
          Native.eval_nif(w, big)
          acc
        rescue
          RuntimeError -> acc + 1
        end
      end)

    assert rejected > 0
    Native.stop_worker(w)
  end

  test "dropping many busy streams reaps cleanly without crashing or leaking" do
    Native.clear_cache()
    base = Native.get_active_memory()

    for i <- 1..100 do
      w = Native.create_worker(32)
      a = Native.random_normal(w, [256, 256], {:f, 32}, 0.0, 1.0, nil)
      b = Native.matmul(w, a, a)
      for _ <- 1..3, do: Native.eval_nif(w, b)
      # Half are closed explicitly; half are dropped for GC — both paths
      # hand the thread to the reaper.
      if rem(i, 2) == 0, do: Native.stop_worker(w)
    end

    :erlang.garbage_collect()
    Process.sleep(200)
    Native.clear_cache()

    # Still responsive (no reaper deadlock/crash) and memory is reclaimed.
    assert Native.get_active_memory() <= base + 64_000_000
  end

  test "Emily.Stream.close/1 stops the worker; later use raises" do
    stream = Emily.Stream.new(:gpu)

    # Sanity: the stream works before closing.
    six =
      Emily.Stream.with_stream(stream, fn ->
        Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend) |> Nx.sum() |> Nx.to_number()
      end)

    assert six == 6.0

    assert :ok = Emily.Stream.close(stream)

    assert_raise RuntimeError, fn ->
      Emily.Stream.with_stream(stream, fn ->
        Nx.add(Nx.tensor([1.0], backend: Emily.Backend), 1.0) |> Nx.to_number()
      end)
    end
  end
end
