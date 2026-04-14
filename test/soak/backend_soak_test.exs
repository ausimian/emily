defmodule Emily.Soak.BackendTest do
  @moduledoc """
  Memory-stability soak test for the Backend layer. The Native layer
  already has its own soak (`test/soak/memory_test.exs`); this one
  exercises the Nx-level call path so we catch leaks introduced by the
  Backend wrappers (ref/wrap helpers, BinaryBackend fallbacks, etc.).

  Design mirrors the Native soak:
    * Small workload (a 2-layer MLP forward pass on a 256×256 tensor).
    * Warm-up so MLX allocator setup doesn't show as phantom growth.
    * Many iterations so per-iter leaks dominate arena drift.
    * Tolerance sized to a handful of retained tensors.
  """

  use ExUnit.Case, async: false

  alias Emily.Native

  @moduletag :soak

  @dim 256
  @iters 500
  @warmup 5

  # With a ~256 KB working set per iter and a per-iter leak rate of 1%
  # we'd grow by ~1.2 MB; a 4 MB tolerance catches rates above ~3%.
  @tolerance_bytes 4 * 1024 * 1024

  defp workload(w1, w2, x) do
    x
    |> Nx.dot(w1)
    |> Nx.max(0.0)
    |> Nx.dot(w2)
    |> Nx.sum()
    |> Nx.to_number()
  end

  test "10k backend forward passes return to baseline" do
    backend = Emily.Backend

    w1 = Nx.iota({@dim, @dim}, type: {:f, 32}, backend: backend) |> Nx.divide(@dim * @dim)
    w2 = Nx.iota({@dim, @dim}, type: {:f, 32}, backend: backend) |> Nx.divide(@dim * @dim)
    x = Nx.iota({@dim, @dim}, type: {:f, 32}, backend: backend) |> Nx.divide(@dim * @dim)

    for _ <- 1..@warmup, do: workload(w1, w2, x)

    :erlang.garbage_collect()
    Native.clear_cache()
    baseline = Native.get_active_memory()
    Native.reset_peak_memory()

    for _ <- 1..@iters, do: workload(w1, w2, x)

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
           """
  end
end
