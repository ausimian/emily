defmodule Emily.CompilerEquivalenceTest do
  @moduledoc """
  CM1 compiler-equivalence — the PRIMARY correctness gate for the
  Expr->MLX single-NIF compiler.

  Every function is run two ways and asserted **bit-identical**:

    * `compiler: Emily.Compiler, native: true` — the new path: the
      Nx.Defn.Expr is lowered to a flat IR, compiled into one `Program`,
      and replayed in a single NIF call.
    * `compiler: Emily.Compiler` — the existing path: `Nx.Defn.Evaluator`
      walking the same Expr op-by-op through `Emily.Backend`.

  Both ultimately call the same `mlx::core::*` ops in the same order, so
  matching values is exact (not tolerance-based). The oracle is "the
  backend in non-defn mode", per Emily's testing philosophy.
  """
  use ExUnit.Case, async: true

  @native [compiler: Emily.Compiler, native: true]
  @eval [compiler: Emily.Compiler]

  defp et(data, opts \\ []), do: Nx.tensor(data, [backend: Emily.Backend] ++ opts)

  defp run(fun, args, opts), do: apply(Nx.Defn.jit(fun, opts), args)

  # Assert the native single-NIF path matches the Evaluator path
  # bit-for-bit on a single-tensor output.
  defp assert_equiv(fun, args) do
    native = run(fun, args, @native)
    eval = run(fun, args, @eval)

    assert %Emily.Backend{} = native.data
    assert native.type == eval.type
    assert native.shape == eval.shape

    assert Nx.to_binary(native) == Nx.to_binary(eval),
           "native vs evaluator mismatch\n  native: #{inspect(Nx.to_flat_list(native))}\n  eval:   #{inspect(Nx.to_flat_list(eval))}"

    native
  end

  defp assert_in_delta_list(a, b, tol) do
    assert length(a) == length(b)
    Enum.zip(a, b) |> Enum.each(fn {x, y} -> assert_in_delta(x, y, tol) end)
  end

  describe "unary elementwise" do
    test "float unary ops match the evaluator" do
      x = et([0.5, -1.25, 2.0, 0.1])

      for op <- [
            &Nx.exp/1,
            &Nx.tanh/1,
            &Nx.sigmoid/1,
            &Nx.negate/1,
            &Nx.abs/1,
            &Nx.sqrt/1,
            &Nx.rsqrt/1,
            &Nx.sign/1,
            &Nx.floor/1,
            &Nx.ceil/1,
            &Nx.sin/1,
            &Nx.cos/1,
            &Nx.log/1,
            &Nx.erf/1
          ] do
        assert_equiv(op, [x])
      end
    end
  end

  describe "binary arithmetic" do
    test "elementwise arithmetic matches" do
      a = et([1.0, 2.0, 3.0, 4.0])
      b = et([10.0, 20.0, 30.0, 40.0])

      assert_equiv(&Nx.add/2, [a, b])
      assert_equiv(&Nx.subtract/2, [a, b])
      assert_equiv(&Nx.multiply/2, [a, b])
      assert_equiv(&Nx.divide/2, [a, b])
      assert_equiv(&Nx.pow/2, [a, b])
      assert_equiv(&Nx.max/2, [a, b])
      assert_equiv(&Nx.min/2, [a, b])
    end

    test "broadcasting matches" do
      a = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = et([10.0, 20.0, 30.0])
      assert_equiv(&Nx.add/2, [a, b])
    end

    test "mixed-dtype add coerces to out.type like the backend" do
      a = et([1, 2, 3], type: :s32)
      b = et([0.5, 0.5, 0.5], type: :f32)
      assert_equiv(&Nx.add/2, [a, b])
    end

    test "scalar constant operand (materialized capture)" do
      x = et([1.0, 2.0, 3.0])
      assert_equiv(fn t -> Nx.add(t, 1.5) end, [x])
      assert_equiv(fn t -> Nx.multiply(t, 2.0) end, [x])
    end

    test "embedded tensor literal lowers to a capture" do
      x = et([1.0, 2.0, 3.0])
      assert_equiv(fn t -> Nx.add(t, Nx.tensor([10.0, 20.0, 30.0])) end, [x])
    end
  end

  describe "compare / logical (bool -> u8 coercion)" do
    test "comparisons match and produce u8" do
      a = et([1.0, 2.0, 3.0])
      b = et([2.0, 2.0, 2.0])

      for op <- [&Nx.equal/2, &Nx.not_equal/2, &Nx.less/2, &Nx.greater/2, &Nx.greater_equal/2] do
        out = assert_equiv(op, [a, b])
        assert out.type == {:u, 8}
      end
    end
  end

  describe "cast / shape" do
    test "as_type matches" do
      x = et([1.7, -2.3, 3.9])
      assert_equiv(fn t -> Nx.as_type(t, :s32) end, [x])
      assert_equiv(fn t -> Nx.as_type(t, :bf16) end, [x])
    end

    test "reshape / transpose / squeeze match" do
      x = et([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      assert_equiv(fn t -> Nx.reshape(t, {2, 3}) end, [x])
      assert_equiv(fn t -> t |> Nx.reshape({2, 3}) |> Nx.transpose() end, [x])

      y = et([[[1.0], [2.0], [3.0]]])
      assert_equiv(fn t -> Nx.squeeze(t) end, [y])
    end

    test "broadcast matches" do
      x = et([1.0, 2.0, 3.0])
      assert_equiv(fn t -> Nx.broadcast(t, {2, 3}) end, [x])
    end
  end

  describe "dot / matmul" do
    test "2D @ 2D matmul matches" do
      a = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = et([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      assert_equiv(&Nx.dot/2, [a, b])
    end

    test "vector dot product matches" do
      a = et([1.0, 2.0, 3.0])
      b = et([4.0, 5.0, 6.0])
      assert_equiv(&Nx.dot/2, [a, b])
    end

    test "tensordot over explicit contraction axes matches" do
      a = et([[1.0, 2.0], [3.0, 4.0]])
      b = et([[5.0, 6.0], [7.0, 8.0]])
      assert_equiv(fn x, y -> Nx.dot(x, [1], [], y, [0], []) end, [a, b])
    end

    test "batched dot (3-D) matches" do
      a = et([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      b = et([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
      assert_equiv(fn x, y -> Nx.dot(x, [2], [0], y, [1], [0]) end, [a, b])
    end
  end

  describe "reductions" do
    test "sum over all axes and specific axes" do
      x = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equiv(fn t -> Nx.sum(t) end, [x])
      assert_equiv(fn t -> Nx.sum(t, axes: [0]) end, [x])
      assert_equiv(fn t -> Nx.sum(t, axes: [1]) end, [x])
      assert_equiv(fn t -> Nx.sum(t, axes: [1], keep_axes: true) end, [x])
    end

    test "product / reduce_max / reduce_min" do
      x = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equiv(fn t -> Nx.product(t, axes: [1]) end, [x])
      assert_equiv(fn t -> Nx.reduce_max(t, axes: [0]) end, [x])
      assert_equiv(fn t -> Nx.reduce_min(t) end, [x])
    end

    test "softmax-style exp/sum/divide composite matches" do
      x = et([1.0, 2.0, 3.0, 4.0])

      assert_equiv(
        fn t ->
          e = Nx.exp(t)
          Nx.divide(e, Nx.sum(e))
        end,
        [x]
      )
    end
  end

  describe "select / slice / iota" do
    test "select / where matches" do
      a = et([1.0, 2.0, 3.0, 4.0])
      b = et([10.0, 20.0, 30.0, 40.0])
      assert_equiv(fn x, y -> Nx.select(Nx.greater(x, 2.0), x, y) end, [a, b])
    end

    test "relu via select matches" do
      x = et([-1.0, 0.5, -2.0, 3.0])
      assert_equiv(fn t -> Nx.max(t, 0.0) end, [x])

      assert_equiv(fn t -> Nx.select(Nx.greater(t, 0.0), t, Nx.broadcast(0.0, Nx.shape(t))) end, [
        x
      ])
    end

    test "static slice matches" do
      x = et([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
      assert_equiv(fn t -> Nx.slice(t, [0, 1], [2, 2]) end, [x])
      assert_equiv(fn t -> t[[0..1, 1..2]] end, [x])
    end

    test "iota lowers to a constant and matches" do
      x = et([0.0, 0.0, 0.0, 0.0])
      assert_equiv(fn t -> Nx.add(t, Nx.iota({4})) end, [x])
      assert_equiv(fn t -> Nx.add(t, Nx.iota({4}, type: :f32)) end, [x])
    end
  end

  describe "fused kernels (Emily.Fast blocks)" do
    test "rms_norm matches the fused kernel via the Evaluator" do
      x = et([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 2.0, -2.0]])
      w = et([1.0, 1.0, 1.0, 1.0])
      assert_equiv(fn x, w -> Emily.Fast.rms_norm(x, w) end, [x, w])
      assert_equiv(fn x, w -> Emily.Fast.rms_norm(x, w, eps: 1.0e-5) end, [x, w])
    end

    test "layer_norm matches the fused kernel via the Evaluator" do
      x = et([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 2.0, -2.0]])
      w = et([1.0, 0.5, 1.0, 0.5])
      b = et([0.0, 0.1, 0.0, -0.1])
      assert_equiv(fn x, w, b -> Emily.Fast.layer_norm(x, w, b) end, [x, w, b])
    end

    test "rms_norm composed inside a larger graph" do
      x = et([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 2.0, -2.0]])
      w = et([1.0, 1.0, 1.0, 1.0])

      assert_equiv(
        fn x, w ->
          x
          |> Emily.Fast.rms_norm(w)
          |> Nx.multiply(2.0)
          |> Nx.add(1.0)
        end,
        [x, w]
      )
    end

    test "rope matches the fused kernel" do
      # {batch, heads, seq, head_dim}
      x = Nx.iota({1, 2, 4, 8}, type: :f32, backend: Emily.Backend) |> Nx.divide(10.0)
      offset = Nx.tensor(0, type: :s32, backend: Emily.Backend)

      assert_equiv(fn x, off -> Emily.Fast.rope(x, off, dims: 8) end, [x, offset])
      assert_equiv(fn x, off -> Emily.Fast.rope(x, off, dims: 8, base: 5000.0) end, [x, offset])
    end

    test "sdpa matches the fused kernel (causal and non-causal)" do
      shape = {1, 2, 4, 8}
      q = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      k = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      v = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(32.0)

      assert_equiv(fn q, k, v -> Emily.Fast.scaled_dot_product_attention(q, k, v) end, [q, k, v])

      assert_equiv(
        fn q, k, v -> Emily.Fast.scaled_dot_product_attention(q, k, v, causal: true) end,
        [q, k, v]
      )
    end

    test "sdpa with an additive mask matches the fused kernel" do
      shape = {1, 2, 4, 8}
      q = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      k = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      v = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(32.0)
      mask = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {1, 1, 4, 4})

      assert_equiv(
        fn q, k, v, m -> Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m) end,
        [q, k, v, mask]
      )
    end
  end

  describe "quantized matmul block" do
    test "affine int4 quantized_matmul_defn matches the Evaluator and the eager kernel" do
      # weight {out=4, in=64}, transpose: true (MLX/from_dense default).
      w = Nx.iota({4, 64}, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      qw = Emily.QuantizedWeight.from_dense(w, group_size: 64, bits: 4)
      x = Nx.iota({2, 64}, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)

      # The QuantizedWeight (an Nx.Container) is passed as an argument so
      # its tensors flow in as Expr parameters (device refs, no host copy).
      f = fn x, qw -> Emily.Quantization.quantized_matmul_defn(x, qw) end

      native = run(f, [x, qw], @native)
      eval = run(f, [x, qw], @eval)

      assert native.shape == {2, 4}
      # Native single-NIF path vs Evaluator — both the fused kernel.
      assert Nx.to_binary(native) == Nx.to_binary(eval)

      # And it agrees with the eager quantized_matmul/2 (same kernel).
      eager = Emily.Quantization.quantized_matmul(x, qw)
      assert_in_delta_list(Nx.to_flat_list(native), Nx.to_flat_list(eager), 1.0e-4)
    end

    test "affine int8 quantized_matmul_defn matches the Evaluator" do
      w = Nx.iota({3, 64}, type: :f32, backend: Emily.Backend) |> Nx.divide(96.0)
      qw = Emily.QuantizedWeight.from_dense(w, group_size: 64, bits: 8)
      x = Nx.iota({2, 64}, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)

      assert_equiv(fn x, qw -> Emily.Quantization.quantized_matmul_defn(x, qw) end, [x, qw])
    end
  end

  describe "concatenate / conv" do
    test "concatenate along axes matches" do
      a = et([[1.0, 2.0], [3.0, 4.0]])
      b = et([[5.0, 6.0], [7.0, 8.0]])
      assert_equiv(fn a, b -> Nx.concatenate([a, b], axis: 0) end, [a, b])
      assert_equiv(fn a, b -> Nx.concatenate([a, b], axis: 1) end, [a, b])
      assert_equiv(fn a, b -> Nx.concatenate([a, b, a]) end, [a, b])
    end

    test "2-D conv (patch-embed style) matches the evaluator" do
      # NCHW input {1, 3, 8, 8}; OIHW kernel {4, 3, 2, 2}, stride 2 (patches).
      x = Nx.iota({1, 3, 8, 8}, type: :f32, backend: Emily.Backend) |> Nx.divide(192.0)
      k = Nx.iota({4, 3, 2, 2}, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      assert_equiv(fn x, k -> Nx.conv(x, k, strides: [2, 2]) end, [x, k])
    end

    test "conv with padding + feature groups matches" do
      x = Nx.iota({1, 4, 6, 6}, type: :f32, backend: Emily.Backend) |> Nx.divide(144.0)
      k = Nx.iota({4, 2, 3, 3}, type: :f32, backend: Emily.Backend) |> Nx.divide(72.0)

      assert_equiv(
        fn x, k ->
          Nx.conv(x, k, strides: [1, 1], padding: [{1, 1}, {1, 1}], feature_group_size: 2)
        end,
        [x, k]
      )
    end
  end

  describe "take (embedding lookup)" do
    test "take matches the Evaluator" do
      embed = et([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      ids = Nx.tensor([0, 3, 1, 0, 2], type: :s32, backend: Emily.Backend)
      assert_equiv(fn e, i -> Nx.take(e, i) end, [embed, ids])
    end

    test "embedding lookup + scale (Gemma-style) matches" do
      embed = et([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
      ids = Nx.tensor([2, 0, 1], type: :s32, backend: Emily.Backend)
      assert_equiv(fn e, i -> Nx.take(e, i) |> Nx.multiply(2.5) end, [embed, ids])
    end
  end

  describe "take_along_axis" do
    test "along the last axis matches the Evaluator" do
      x = et([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
      idx = Nx.tensor([[3, 0], [1, 2], [0, 3]], type: :s64, backend: Emily.Backend)
      assert_equiv(fn t, i -> Nx.take_along_axis(t, i, axis: 1) end, [x, idx])
    end

    test "along axis 0 matches" do
      x = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      idx = Nx.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 1]], type: :s64, backend: Emily.Backend)
      assert_equiv(fn t, i -> Nx.take_along_axis(t, i, axis: 0) end, [x, idx])
    end

    test "3-D gather along the last axis (transformer-shaped) matches" do
      x = Nx.iota({1, 2, 4}, type: :f32, backend: Emily.Backend) |> Nx.divide(8.0)
      idx = Nx.tensor([[[3, 1], [0, 2]]], type: :s32, backend: Emily.Backend)
      assert_equiv(fn t, i -> Nx.take_along_axis(t, i, axis: 2) end, [x, idx])
    end
  end

  describe "dynamic put_slice (KV-cache write)" do
    test "put_slice at a runtime offset matches the Evaluator" do
      # {batch, n_kv_heads, max_len, head_dim} KV buffer; write one token.
      buf = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {1, 2, 6, 4})
      upd = Nx.iota({1, 2, 1, 4}, type: :f32, backend: Emily.Backend) |> Nx.divide(10.0)
      offset = Nx.tensor(3, type: :s32, backend: Emily.Backend)

      assert_equiv(
        fn buf, upd, off -> Nx.put_slice(buf, [0, 0, off, 0], upd) end,
        [buf, upd, offset]
      )
    end

    test "put_slice with all-static starts matches" do
      buf = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {4, 4})
      upd = Nx.iota({2, 2}, type: :f32, backend: Emily.Backend) |> Nx.add(1.0)
      assert_equiv(fn buf, upd -> Nx.put_slice(buf, [1, 1], upd) end, [buf, upd])
    end
  end

  describe "two-layer MLP forward (matmul-dominated)" do
    test "matches the evaluator end-to-end" do
      x = et([[0.1, 0.2, 0.3, 0.4]])
      w1 = et([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
      b1 = et([0.01, 0.02])
      w2 = et([[0.5], [0.6]])
      b2 = et([0.001])

      assert_equiv(
        fn x, w1, b1, w2, b2 ->
          x
          |> Nx.dot(w1)
          |> Nx.add(b1)
          |> Nx.tanh()
          |> Nx.dot(w2)
          |> Nx.add(b2)
          |> Nx.sigmoid()
        end,
        [x, w1, b1, w2, b2]
      )
    end
  end

  describe "composite graphs" do
    test "elementwise MLP-style block (no matmul)" do
      x = et([0.1, -0.2, 0.3, -0.4])
      w = et([0.5, 0.5, 0.5, 0.5])
      b = et([1.0, 1.0, 1.0, 1.0])

      assert_equiv(fn x, w, b -> x |> Nx.multiply(w) |> Nx.add(b) |> Nx.sigmoid() end, [x, w, b])
    end

    test "shared subexpression is computed once (DAG, not tree)" do
      x = et([1.0, 2.0, 3.0])

      assert_equiv(
        fn t ->
          s = Nx.exp(t)
          Nx.add(s, s)
        end,
        [x]
      )
    end

    test "tuple output reassembles correctly" do
      x = et([1.0, -2.0, 3.0])
      native = run(fn t -> {Nx.exp(t), Nx.negate(t)} end, [x], @native)
      eval = run(fn t -> {Nx.exp(t), Nx.negate(t)} end, [x], @eval)

      assert {n1, n2} = native
      assert {e1, e2} = eval
      assert Nx.to_binary(n1) == Nx.to_binary(e1)
      assert Nx.to_binary(n2) == Nx.to_binary(e2)
    end

    test "identity (output is a parameter) round-trips" do
      x = et([1.0, 2.0, 3.0])
      assert_equiv(fn t -> t end, [x])
    end
  end

  describe "reuse" do
    test "a compiled closure runs many inputs of the same signature" do
      f = Nx.Defn.jit(fn t -> Nx.tanh(t) end, @native)

      for data <- [[0.0, 1.0], [2.0, 3.0], [-1.0, -2.0]] do
        x = et(data)
        assert Nx.to_binary(f.(x)) == Nx.to_binary(Nx.tanh(x))
      end
    end
  end
end
