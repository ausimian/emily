defmodule Emily.Soak.MemoryTest do
  @moduledoc """
  Memory-stability soak test. Tagged `:soak` for discoverability, but
  runs as part of the default suite — the check completes in ~1 s.

  Design (the *why* matters — the first cut was effectively a no-op
  and the second couldn't tell a leak from Metal arena drift):

    * Each iteration allocates a 1 MB tensor, evaluates it, drops it.
    * We run many iterations (2000) so MLX's constant arena overhead
      (roughly 2 tensors' worth persisted past `clear_cache`) becomes
      negligible versus any per-iteration leak — a per-iter leak
      accumulates linearly, arena drift does not.
    * Baseline is sampled *after* a warm-up loop so MLX's one-time
      allocator setup (kernel caches, Metal libraries, lookup tables)
      doesn't show up as phantom growth.
    * Tolerance is sized so the observed ~2 MB of Metal arena drift
      passes, but the test fails once the cumulative leak exceeds
      ~2 retained tensors total — i.e. a leak rate above ~0.1 %.

  `Native.to_binary/1` returns a resource binary aliasing MLX memory.
  The ProcBin is tiny (~64 B) so BEAM's binary-vheap GC doesn't fire
  based on the external data size; resource binaries can accumulate
  and pin MLX buffers. We GC every @gc_every iterations so the test
  reflects steady-state usage, not transient accumulation.

  A tight tolerance matters — a generous one makes the test pass even
  when refcounts are broken.
  """

  use ExUnit.Case, async: false

  alias Emily.Native

  @moduletag :soak

  # 256 Ki × f32 = 1 MB per iter. Small so the workload fits in ~1 s
  # of wall time even at high iteration counts.
  @tensor_elems 256 * 1024
  @iters 2_000
  @warmup 20

  # See moduledoc: periodic GC keeps to_binary's aliased resource
  # binaries from accumulating. Every 10 iters ≈ 10 MB of MLX-pinned
  # memory at the peak, well within the tolerance window.
  @gc_every 10

  # Measured drift under the full suite is ~2 MB (Metal pool keeps
  # ~2 tensors live past clear_cache). 4 MB tolerance absorbs that
  # without masking a leak of more than a couple of retained tensors;
  # a 1 % per-iter leak rate would blow through it by ~5×.
  @tolerance_bytes 4 * 1024 * 1024

  defp workload(data) do
    w = Emily.MlxStream.worker(Emily.MlxStream.Default)
    a = Native.from_binary(data, [@tensor_elems], {:f, 32})
    b = Native.multiply(w, a, a)
    _ = Native.to_binary(w, b)
    :ok
  end

  test "repeated 1 MB allocate/eval/drop returns to baseline" do
    # Build the source binary once; we're exercising MLX's allocator,
    # not binary copy churn on the BEAM side.
    data = for _ <- 1..@tensor_elems, into: <<>>, do: <<1.0::float-32-native>>

    for _ <- 1..@warmup, do: workload(data)

    :erlang.garbage_collect()
    Native.clear_cache()
    baseline = Native.get_active_memory()
    Native.reset_peak_memory()

    for i <- 1..@iters do
      workload(data)
      if rem(i, @gc_every) == 0, do: :erlang.garbage_collect()
    end

    :erlang.garbage_collect()
    Native.clear_cache()

    final = Native.get_active_memory()
    peak = Native.get_peak_memory()
    delta = final - baseline

    assert delta <= @tolerance_bytes,
           """
           active-memory delta #{delta} bytes exceeds tolerance #{@tolerance_bytes}
             baseline:       #{baseline}
             final:          #{final}
             peak:           #{peak}
             iters:          #{@iters}
             bytes/tensor:   #{@tensor_elems * 4}
           A single retained tensor per iteration would grow delta by
           #{@iters * @tensor_elems * 4} bytes (~#{div(@iters * @tensor_elems * 4, 1024 * 1024)} MB).
           """
  end
end
