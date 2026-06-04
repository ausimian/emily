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
