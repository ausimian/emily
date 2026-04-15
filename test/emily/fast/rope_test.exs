defmodule Emily.Fast.RoPETest do
  @moduledoc """
  Tests for `Emily.Fast.rope/3` and `Emily.Fast.rope_with_freqs/4`.

  The input convention here matches MLX's fast rope — x has shape
  `{..., seq, head_dim}`. For real transformer use, the shim places
  the heads axis ahead of seq; that's a shape permutation the caller
  handles, not something rope itself cares about.
  """

  use ExUnit.Case, async: false

  import Emily.BackendGenerators, only: [assert_close: 3]

  @f32_tol 1.0e-4

  setup do
    prev = Nx.default_backend()
    Nx.default_backend(Emily.Backend)
    on_exit(fn -> Nx.default_backend(prev) end)
    :ok
  end

  describe "rope/3 (standard theta)" do
    test "fused matches defn fallback for split-half layout" do
      dims = 32
      shape = {2, 4, dims}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
      ref_offset = Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      offset = Nx.backend_copy(ref_offset, Emily.Backend)

      fun = fn x, offset ->
        Emily.Fast.rope(x, offset, dims: dims, traditional: false, base: 10_000.0)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_offset)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, offset)

      assert_close(fused, expected, tol: @f32_tol)
    end

    test "fused matches defn fallback for traditional (interleave) layout" do
      dims = 16
      shape = {1, 4, dims}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(50)
      ref_offset = Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      offset = Nx.backend_copy(ref_offset, Emily.Backend)

      fun = fn x, offset ->
        Emily.Fast.rope(x, offset, dims: dims, traditional: true, base: 10_000.0)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_offset)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, offset)

      assert_close(fused, expected, tol: @f32_tol)
    end

    test "honours non-zero offset (KV cache)" do
      dims = 16
      shape = {1, 3, dims}

      # Slot the new 3 tokens at positions 8, 9, 10 — what the decoder
      # would do after 8 tokens have already been processed.
      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(50)
      ref_offset = Nx.tensor(8, type: :s32, backend: Nx.BinaryBackend)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      offset = Nx.backend_copy(ref_offset, Emily.Backend)

      fun = fn x, offset ->
        Emily.Fast.rope(x, offset, dims: dims, traditional: false, base: 10_000.0)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_offset)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, offset)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end

  describe "rope_with_freqs/4" do
    # End-to-end correctness on llama3-scaled freqs is covered by the
    # Qwen3 conformance suite. Here we just assert fused == defn fallback
    # so the backend routing + freqs plumbing through the NIF is sound.
    test "fused matches defn fallback under an arbitrary freqs table" do
      dims = 16
      shape = {1, 4, dims}

      # Llama-3-shaped freqs: standard inv_freq divided by a constant
      # factor, mimicking Bumblebee's low/high-frequency rescale.
      ref_inv_freq =
        Nx.iota({div(dims, 2)}, type: :f32, backend: Nx.BinaryBackend)
        |> Nx.multiply(2.0)
        |> Nx.divide(dims)
        |> then(&Nx.divide(1.0, Nx.pow(10_000.0, &1)))
        |> Nx.divide(1.37)

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
      ref_offset = Nx.tensor(0, type: :s32, backend: Nx.BinaryBackend)

      expected =
        Nx.Defn.jit(
          fn x, offset, freqs ->
            Emily.Fast.rope_with_freqs(x, offset, freqs, dims: dims, traditional: false)
          end,
          compiler: Nx.Defn.Evaluator
        ).(ref_x, ref_offset, ref_inv_freq)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      offset = Nx.backend_copy(ref_offset, Emily.Backend)
      inv_freq = Nx.backend_copy(ref_inv_freq, Emily.Backend)

      fused =
        Nx.Defn.jit(
          fn x, offset, freqs ->
            Emily.Fast.rope_with_freqs(x, offset, freqs, dims: dims, traditional: false)
          end,
          compiler: Emily.Compiler
        ).(x, offset, inv_freq)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end
end
