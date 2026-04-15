defmodule Emily.Soak.QuantizedMemoryTest do
  @moduledoc """
  Memory-stability soak for the quantized-matmul path.

  Why a separate soak: quantized inference is allocator-pattern different
  from fp16. Packed weights load once and never re-quantize, so the
  working set for a forward pass is `x + packed w + scales/biases +
  output` — smaller than fp16 by roughly the bits ratio. A leaking
  refcount on any of the quantize outputs (w_q, scales, biases) would
  still accumulate linearly across iterations, just at a smaller per-iter
  byte budget than the fp16 soak catches.

  Structure mirrors `soak/memory_test.exs`: warm up, sample baseline
  after `clear_cache`, run many iterations, assert active memory returns
  within a tight tolerance.
  """

  use ExUnit.Case, async: false

  alias Emily.Native
  alias Emily.Quantization
  alias Emily.QuantizedWeight

  @moduletag :soak

  # Weight stays alive for the whole test — we're measuring per-iter
  # leaks of activation/output allocations, not weight reload.
  @out_feat 64
  @in_feat 1024
  @batch 8
  @iters 1_000
  @warmup 20

  # Activation + output per iter:
  #   x:   8 × 1024 × 4 bytes = 32 KiB
  #   out: 8 × 64 × 4 bytes   = 2 KiB
  # Weight packed (int4, group_size=64):
  #   value: 64 × 128 × 4      = 32 KiB
  #   scales/biases: 64×16×4×2 = 8 KiB
  # A leak of one retained activation per iter would blow past this by 10×
  # within the iteration count; Metal arena drift is bounded around ~2 MB.
  @tolerance_bytes 4 * 1024 * 1024

  defp workload(x, qw) do
    y = Quantization.quantized_matmul(x, qw)
    # Force materialization so MLX actually executes the kernel; otherwise
    # the lazy graph would just compose nodes without hitting the
    # allocator the way a real forward pass does.
    %Nx.Tensor{data: %Emily.Backend{ref: r}} = y
    :ok = Native.eval(r)
    :ok
  end

  test "repeated quantized_matmul returns to baseline" do
    # Fresh quantized weight. The f32 dense tensor is dropped after
    # from_dense/2; the packed value/scales/biases live for the test.
    w =
      Nx.iota({@out_feat, @in_feat}, backend: Emily.Backend, type: :f32)
      |> Nx.divide(@out_feat * @in_feat / 2)
      |> Nx.subtract(1.0)

    qw = QuantizedWeight.from_dense(w)

    x =
      Nx.iota({@batch, @in_feat}, backend: Emily.Backend, type: :f32)
      |> Nx.divide(@in_feat)
      |> Nx.subtract(0.5)

    for _ <- 1..@warmup, do: workload(x, qw)

    :erlang.garbage_collect()
    Native.clear_cache()
    baseline = Native.get_active_memory()
    Native.reset_peak_memory()

    for _ <- 1..@iters, do: workload(x, qw)

    :erlang.garbage_collect()
    Native.clear_cache()

    final = Native.get_active_memory()
    peak = Native.get_peak_memory()
    delta = final - baseline

    assert delta <= @tolerance_bytes,
           """
           active-memory delta #{delta} bytes exceeds tolerance #{@tolerance_bytes}
             baseline:   #{baseline}
             final:      #{final}
             peak:       #{peak}
             iters:      #{@iters}

           A single retained output per iteration would grow delta by
           #{@iters * @batch * @out_feat * 4} bytes.
           """
  end
end
