defmodule Emily.Fast.RoPETest do
  @moduledoc """
  Tests for `Emily.Fast.rope/3` and `Emily.Fast.rope_with_freqs/4`.

  The input convention here matches MLX's fast rope — x has shape
  `{..., seq, head_dim}`. For real transformer use, the shim places
  the heads axis ahead of seq; that's a shape permutation the caller
  handles, not something rope itself cares about.
  """

  use ExUnit.Case, async: true
  doctest Emily.Fast, only: [rope: 3, rope_with_freqs: 4]

  import Emily.BackendGenerators, only: [assert_close: 3]

  alias Emily.FastFallback

  @f32_tol 1.0e-4

  setup do
    Nx.default_backend(Emily.Backend)
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

      expected = FastFallback.run(fun, [ref_x, ref_offset])
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

      expected = FastFallback.run(fun, [ref_x, ref_offset])
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

      expected = FastFallback.run(fun, [ref_x, ref_offset])
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
        FastFallback.run(
          fn x, offset, freqs ->
            Emily.Fast.rope_with_freqs(x, offset, freqs, dims: dims, traditional: false)
          end,
          [ref_x, ref_offset, ref_inv_freq]
        )

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

  describe "per-batch offset" do
    # A {batch} offset rotates each entry along the leading axis from its
    # own starting position, the same as one call per entry with a
    # scalar offset.
    for {lane, {backend, compiler}} <- [
          fused: {Emily.Backend, [compiler: Emily.Compiler]},
          native:
            {Emily.Backend, [compiler: Emily.Compiler, native: true, native_fallback: :raise]},
          fallback: {Nx.BinaryBackend, [compiler: Nx.Defn.Evaluator]}
        ] do
      test "matches per-entry scalar offsets (#{lane})" do
        Nx.default_backend(unquote(backend))
        dims = 8
        x = Nx.iota({2, 2, 3, dims}, type: :f32) |> Nx.divide(50)
        freqs = Nx.tensor([1.0, 0.1, 0.01, 0.001])

        rope = fn x, offset, _freqs -> Emily.Fast.rope(x, offset, dims: dims) end

        with_freqs = fn x, offset, freqs ->
          Emily.Fast.rope_with_freqs(x, offset, freqs, dims: dims)
        end

        for fun <- [rope, with_freqs] do
          jit = Nx.Defn.jit(fun, unquote(compiler))
          batched = jit.(x, Nx.tensor([3, 7]), freqs)

          per_entry =
            Nx.concatenate([
              jit.(x[0..0], Nx.tensor(3), freqs),
              jit.(x[1..1], Nx.tensor(7), freqs)
            ])

          assert_close(batched, per_entry, tol: @f32_tol)
        end
      end

      test "scales the whole position, offset included (#{lane})" do
        Nx.default_backend(unquote(backend))
        dims = 8
        x = Nx.iota({2, 2, 3, dims}, type: :f32) |> Nx.divide(50)
        offset = Nx.tensor([3, 7])

        actual =
          Nx.Defn.jit(
            fn x, offset -> Emily.Fast.rope(x, offset, dims: dims, scale: 0.5) end,
            unquote(compiler)
          ).(x, offset)

        assert_close(actual, split_half_rope(x, offset, 0.5, dims), tol: @f32_tol)
      end
    end
  end

  # Independent split-half RoPE on Nx.BinaryBackend: position
  # (i + offset[b]) * scale for token i of batch entry b.
  defp split_half_rope(x, offset, scale, dims) do
    x = Nx.backend_copy(x, Nx.BinaryBackend)
    offset = Nx.backend_copy(offset, Nx.BinaryBackend)
    {batch, _heads, seq, _dims} = Nx.shape(x)

    positions =
      Nx.iota({batch, seq}, axis: 1, backend: Nx.BinaryBackend)
      |> Nx.add(Nx.new_axis(offset, 1))
      |> Nx.multiply(scale)

    inv_freq =
      Nx.iota({div(dims, 2)}, type: :f32, backend: Nx.BinaryBackend)
      |> Nx.multiply(2 / dims)
      |> then(&Nx.divide(1.0, Nx.pow(10_000.0, &1)))

    angle = Nx.multiply(Nx.new_axis(positions, -1), inv_freq) |> Nx.new_axis(1)
    {cos, sin} = {Nx.cos(angle), Nx.sin(angle)}

    half = div(dims, 2)
    x1 = Nx.slice_along_axis(x, 0, half, axis: -1)
    x2 = Nx.slice_along_axis(x, half, half, axis: -1)

    Nx.concatenate(
      [
        Nx.subtract(Nx.multiply(x1, cos), Nx.multiply(x2, sin)),
        Nx.add(Nx.multiply(x1, sin), Nx.multiply(x2, cos))
      ],
      axis: -1
    )
  end

  describe "fast_rope_int/8 (integer offset, incremental decode)" do
    # Incremental decode ropes one token at a time at an integer absolute
    # position. It must be correct for seq == 1 — fed the 4D
    # {1, heads, seq, head_dim} layout (in 3D, MLX 0.31 mis-rotates seq == 1).
    # This pins the decode-vs-prefill consistency a generation loop relies on:
    # a single token roped at offset k must equal position k of the full
    # sequence roped at offset 0.
    test "single-token rope at offset k equals position k of a full-sequence rope" do
      dims = 32
      heads = 2
      seq = 8
      k = 5
      w = Emily.MlxStream.default_worker()

      freqs =
        Nx.iota({div(dims, 2)}, type: :f32)
        |> Nx.multiply(2.0)
        |> Nx.divide(dims)
        |> then(&Nx.pow(10_000.0, &1))

      x = Nx.iota({1, heads, seq, dims}, type: :f32) |> Nx.divide(100)

      wrap = fn r, shape ->
        %Nx.Tensor{
          data: %Emily.Backend{ref: r},
          shape: shape,
          type: Emily.Native.dtype(r),
          names: List.duplicate(nil, tuple_size(shape))
        }
      end

      rope_int = fn t, offset, shape ->
        Emily.Native.fast_rope_int(w, t.data.ref, dims, false, nil, 1.0, offset, freqs.data.ref)
        |> wrap.(shape)
      end

      full = rope_int.(x, 0, {1, heads, seq, dims})
      tok = Nx.slice(x, [0, 0, k, 0], [1, heads, 1, dims])
      single = rope_int.(tok, k, {1, heads, 1, dims})

      assert_close(single, Nx.slice(full, [0, 0, k, 0], [1, heads, 1, dims]), tol: @f32_tol)
    end
  end
end
