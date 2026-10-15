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
  alias Emily.Quantization.Transform
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
    :ok = Native.eval(Emily.MlxStream.worker(Emily.MlxStream.Default), r)
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

  describe "Transform.quantize/3" do
    # bf16 [4096, 4096] kernel = 32 MiB. int4 at group_size 64 keeps
    # 8 MiB of packed values plus 1 MiB of bf16 scales and biases.
    @units 4096
    @source_bytes @units * @units * 2

    test "frees the dense source kernels once the caller drops them" do
      :erlang.garbage_collect()
      Native.clear_cache()
      baseline = Native.get_active_memory()

      qstate = quantize_and_drop_source()

      # A pinned source alone would add @source_bytes on top of the
      # quantized weights.
      delta = settled_delta(baseline, div(@source_bytes, 2))

      assert delta < div(@source_bytes, 2),
             """
             active-memory delta #{delta} bytes after quantize/3; the
             #{@source_bytes}-byte dense source kernel is still resident.
             """

      assert %QuantizedWeight{} = qstate.data["fc"]["kernel"]
    end

    # MLX releases a dropped array's buffer a few milliseconds after the
    # BEAM collects its last reference, so poll until the active-memory
    # delta drops below `limit` or 2 s pass. Returns the last delta seen.
    defp settled_delta(baseline, limit, attempts \\ 40) do
      :erlang.garbage_collect()
      Native.clear_cache()
      delta = Native.get_active_memory() - baseline

      if delta < limit or attempts == 0 do
        delta
      else
        Process.sleep(50)
        settled_delta(baseline, limit, attempts - 1)
      end
    end

    # Builds the dense model state, quantizes it and returns only the
    # quantized state, so nothing outside MLX still references the
    # dense kernel once this returns.
    defp quantize_and_drop_source do
      model =
        Axon.input("features", shape: {nil, @units})
        |> Axon.dense(@units, name: "fc")

      kernel = Nx.iota({@units, @units}, backend: Emily.Backend, type: :bf16)
      bias = Nx.broadcast(Nx.tensor(0, backend: Emily.Backend, type: :bf16), {@units})

      # Materialize the kernel; a lazy iota would pin no memory at all.
      :ok = Emily.eval(kernel.data.ref)

      state = Axon.ModelState.new(%{"fc" => %{"kernel" => kernel, "bias" => bias}})
      {_qmodel, qstate} = Transform.quantize(model, state, bits: 4, group_size: 64)
      qstate
    end
  end
end
