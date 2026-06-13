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

    # Direct-mapped unary ops added alongside the Expr op-coverage sweep
    # (see #188). Each routes through the same `mx::*` primitive as the
    # eager unary NIF, so the native single-NIF path is bit-identical to
    # the Evaluator (which dispatches the same op via Emily.Backend).
    test "extra direct-mapped float unary ops match the evaluator" do
      x = et([0.5, -1.25, 2.0, 0.1])

      for op <- [
            &Nx.expm1/1,
            &Nx.tan/1,
            &Nx.sinh/1,
            &Nx.cosh/1,
            &Nx.atan/1,
            &Nx.asinh/1
          ] do
        assert_equiv(op, [x])
      end
    end

    test "inverse-trig unary ops match the evaluator (domain-restricted inputs)" do
      # asin/acos: |x| <= 1; atanh: |x| < 1; acosh: x >= 1.
      x = et([0.1, -0.5, 0.25, -0.75])

      for op <- [&Nx.asin/1, &Nx.acos/1, &Nx.atanh/1] do
        assert_equiv(op, [x])
      end

      assert_equiv(&Nx.acosh/1, [et([1.0, 1.5, 2.0, 3.0])])
    end

    test "round (away from zero) matches the evaluator on tie-breaking inputs" do
      # Nx.round/1 documents "round away from zero"; MLX's mx::round
      # decimals=0 follows the same. The Backend hard-codes decimals=0
      # and the IR dispatcher does too.
      assert_equiv(&Nx.round/1, [et([-1.5, -0.5, 0.5, 1.5, 2.3, -2.7])])
    end

    test "composed erfc and cbrt match the evaluator (no MLX primitive)" do
      # Both compose from existing ops (erfc: 1 - erf; cbrt: sign * abs^(1/3))
      # the same way Emily.Backend does, so the bit pattern stays exact.
      assert_equiv(&Nx.erfc/1, [et([0.0, 0.5, 1.0, -0.5, 2.0])])
      assert_equiv(&Nx.cbrt/1, [et([8.0, -8.0, 0.125, -0.125, 0.0])])
    end

    test "bitwise_not on integer inputs matches the evaluator" do
      for type <- [:s32, :s64, :u8, :u32] do
        assert_equiv(&Nx.bitwise_not/1, [et([0, 1, 2, 255], type: type)])
      end
    end

    test "is_nan / is_infinity on float inputs match the evaluator" do
      # nan/inf live alongside finite values so the predicate output (Nx pred ==
      # {:u, 8}) exercises all three branches; the trailing astype coerces
      # MLX's bool to {:u, 8} like every other unary op.
      x = et([1.0, :infinity, :neg_infinity, :nan, 0.0])

      out = assert_equiv(&Nx.is_nan/1, [x])
      assert out.type == {:u, 8}
      assert_equiv(&Nx.is_infinity/1, [x])
    end

    test "complex unary ops (conjugate / real / imag) match the evaluator" do
      x = et([Complex.new(1.0, 2.0), Complex.new(-3.0, 4.0), Complex.new(0.0, -1.0)])

      conj = assert_equiv(&Nx.conjugate/1, [x])
      assert conj.type == {:c, 64}

      re = assert_equiv(&Nx.real/1, [x])
      assert re.type == {:f, 32}

      im = assert_equiv(&Nx.imag/1, [x])
      assert im.type == {:f, 32}
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

    # Binary arithmetic peers added with the Expr op-coverage sweep (#188).
    # atan2 is a direct @arith_binary mapping (atan2: :arctan2); quotient
    # routes through floor_divide the same way Emily.Backend.quotient/3 does.
    test "atan2 matches the evaluator across the four quadrants" do
      # Quadrant-covering pairs (y, x): include the axes so the result
      # lands on the ±0, ±pi, ±pi/2 boundaries the eager Backend produces.
      y = et([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 2.0, -2.0])
      x = et([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0])
      assert_equiv(&Nx.atan2/2, [y, x])
    end

    test "quotient matches the evaluator for signed and unsigned integer dtypes" do
      # Backend casts both operands to out.type then calls floor_divide.
      # Quotient is integer-only in Nx, so we exercise the s32/s64/u8/u32
      # paths; the native lane matches the evaluator bit-for-bit because
      # both end up at the same mx::floor_divide kernel.
      for type <- [:s32, :s64] do
        a = et([7, -7, 10, -10, 4, -4], type: type)
        b = et([2, 2, 3, 3, 5, 5], type: type)
        assert_equiv(&Nx.quotient/2, [a, b])
      end

      for type <- [:u8, :u32] do
        a = et([7, 10, 4, 255], type: type)
        b = et([2, 3, 5, 2], type: type)
        assert_equiv(&Nx.quotient/2, [a, b])
      end
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

    # logical_xor: MLX has no primitive, so both paths run
    # `(a != 0) != (b != 0)` — Emily.Backend.logical_xor/3 eagerly, the
    # IR via a dedicated composite clause that emits the same three
    # not_equal ops. Inputs span the four truth-table corners across
    # float and integer dtypes; the trailing coerce produces {:u, 8}.
    test "logical_xor matches across float and integer dtypes" do
      a = et([1.0, 0.0, 1.0, 0.0])
      b = et([1.0, 1.0, 0.0, 0.0])
      out = assert_equiv(&Nx.logical_xor/2, [a, b])
      assert out.type == {:u, 8}

      ai = et([1, 0, 1, 0], type: :s32)
      bi = et([1, 1, 0, 0], type: :s32)
      assert_equiv(&Nx.logical_xor/2, [ai, bi])
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

    test "reverse along one and multiple axes matches" do
      x = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert_equiv(fn t -> Nx.reverse(t, axes: [1]) end, [x])
      assert_equiv(fn t -> Nx.reverse(t, axes: [0, 1]) end, [x])
      assert_equiv(fn t -> Nx.reverse(t) end, [x])
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

    # eye is a pure creation op like iota — the IR materialises the
    # identity matrix as a captured constant via Nx.eye on the host
    # backend (which handles the rank > 2 batch broadcast itself, matching
    # Emily.Backend.eye/2). The native and evaluator paths both end up
    # comparing against the same materialised bits.
    test "eye lowers to a constant and matches (2-D and batched)" do
      x = et([[0.0, 0.0], [0.0, 0.0]])
      assert_equiv(fn t -> Nx.add(t, Nx.eye(2)) end, [x])
      assert_equiv(fn t -> Nx.add(t, Nx.eye({2, 2}, type: :f32)) end, [x])

      y = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {2, 3, 3})
      assert_equiv(fn t -> Nx.add(t, Nx.eye({2, 3, 3}, type: :f32)) end, [y])
    end
  end

  describe "LinAlg blocks" do
    # Each mirrors the corresponding Emily.Backend.native_* helper and
    # dispatches to the same mx::linalg::* CPU kernel — so the native
    # single-NIF path and the evaluator land on the same bits even
    # though the decomposition itself runs on the CPU stream.

    test "Cholesky on a 2x2 SPD matrix matches the evaluator" do
      a = et([[4.0, 2.0], [2.0, 3.0]])
      assert_equiv(&Nx.LinAlg.cholesky/1, [a])
    end

    test "Cholesky on a 3x3 SPD matrix matches" do
      # SPD: a = LLᵀ for L = [[1,0,0],[2,1,0],[3,4,1]].
      a = et([[1.0, 2.0, 3.0], [2.0, 5.0, 10.0], [3.0, 10.0, 26.0]])
      assert_equiv(&Nx.LinAlg.cholesky/1, [a])
    end

    test "Solve on a 2x2 well-conditioned A matches the evaluator" do
      a = et([[4.0, 1.0], [2.0, 3.0]])
      b = et([6.0, 7.0])
      assert_equiv(&Nx.LinAlg.solve/2, [a, b])
    end

    test "QR (mode :reduced) on a 3x2 matrix matches the evaluator" do
      a = et([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
      {nq, nr} = run(&Nx.LinAlg.qr/1, [a], @native)
      {eq, er} = run(&Nx.LinAlg.qr/1, [a], @eval)

      assert %Emily.Backend{} = nq.data
      assert Nx.to_binary(nq) == Nx.to_binary(eq)
      assert Nx.to_binary(nr) == Nx.to_binary(er)
    end

    test "Eigh on a 2x2 symmetric matrix matches the evaluator" do
      a = et([[2.0, 1.0], [1.0, 2.0]])
      {nv, nve} = run(&Nx.LinAlg.eigh/1, [a], @native)
      {ev, eve} = run(&Nx.LinAlg.eigh/1, [a], @eval)

      assert Nx.to_binary(nv) == Nx.to_binary(ev)
      assert Nx.to_binary(nve) == Nx.to_binary(eve)
    end

    test "LU on a 3x3 matrix matches the evaluator (P matrix, L, U)" do
      a = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
      {np, nl, nu} = run(&Nx.LinAlg.lu/1, [a], @native)
      {ep, el, eu} = run(&Nx.LinAlg.lu/1, [a], @eval)

      assert %Emily.Backend{} = np.data
      assert Nx.to_binary(np) == Nx.to_binary(ep)
      assert Nx.to_binary(nl) == Nx.to_binary(el)
      assert Nx.to_binary(nu) == Nx.to_binary(eu)
    end

    test "SVD (full_matrices: true, default) on a 3x3 matrix matches the evaluator" do
      # SVD is mathematically non-unique up to sign/order of U/V columns
      # (Backend's property tests acknowledge this at backend_test.exs:771);
      # MLX's SVD also drifts by ~1 ULP in S across separate calls even for
      # the same input. Compare with tolerance for everything SVD-related.
      a = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
      {nu, ns, nv} = run(&Nx.LinAlg.svd/1, [a], @native)
      {eu, es, ev} = run(&Nx.LinAlg.svd/1, [a], @eval)

      assert %Emily.Backend{} = nu.data
      assert nu.shape == eu.shape
      assert nv.shape == ev.shape
      assert_in_delta_list(Nx.to_flat_list(ns), Nx.to_flat_list(es), 1.0e-4)
    end

    test "SVD (full_matrices?: false) on a 3x2 thin matrix matches (IR slices U/V)" do
      a = et([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

      {nu, ns, nv} = run(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, [a], @native)
      {eu, es, ev} = run(fn t -> Nx.LinAlg.svd(t, full_matrices?: false) end, [a], @eval)

      # Thin shapes: U = {m, min(m,n)} = {3, 2}; V = {min(m,n), n} = {2, 2}.
      assert nu.shape == {3, 2}
      assert nv.shape == {2, 2}
      assert nu.shape == eu.shape
      assert nv.shape == ev.shape
      assert_in_delta_list(Nx.to_flat_list(ns), Nx.to_flat_list(es), 1.0e-4)
    end

    test "Determinant on 2x2, 3x3, and NxN matrices matches the evaluator" do
      # 2x2 and 3x3 use closed-form expansions inside the block; NxN
      # uses the LU decomposition. All three paths should land on bits
      # identical to the evaluator.
      assert_equiv(&Nx.LinAlg.determinant/1, [et([[1.0, 2.0], [3.0, 4.0]])])

      assert_equiv(&Nx.LinAlg.determinant/1, [
        et([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [1.0, 0.0, 6.0]])
      ])

      assert_equiv(&Nx.LinAlg.determinant/1, [
        et([
          [1.0, 0.0, 0.0, 1.0],
          [0.0, 2.0, 0.0, 1.0],
          [0.0, 0.0, 3.0, 1.0],
          [1.0, 1.0, 1.0, 4.0]
        ])
      ])
    end
  end

  describe "misc Nx.Block lowerings" do
    # Nx.logical_not (Nx.Block.LogicalNot) — emits the IR :logical_not
    # opcode directly, mirroring Emily.Backend.native_logical_not/2. The
    # trailing coerce produces the {:u, 8} predicate dtype.
    test "logical_not matches the evaluator and produces u8" do
      out = assert_equiv(&Nx.logical_not/1, [et([0, 1, 2, 0, 3], type: :s32)])
      assert out.type == {:u, 8}

      assert_equiv(&Nx.logical_not/1, [et([0.0, 1.0, -1.5, 0.0])])
    end

    # Nx.all_close (Nx.Block.AllClose) — the IR composes the same five
    # primitive sequence as Emily.Backend.native_all_close/4, so the
    # output bit pattern is identical to the evaluator. Exercise the
    # close, far, default-tolerance, custom-tolerance, and equal_nan
    # branches.
    test "all_close on close inputs returns 1 (within default tolerance)" do
      a = et([1.0, 2.0, 3.0])
      b = et([1.0 + 1.0e-6, 2.0 + 1.0e-6, 3.0 - 1.0e-6])
      out = assert_equiv(&Nx.all_close/2, [a, b])
      assert out.type == {:u, 8}
    end

    test "all_close on far inputs returns 0" do
      a = et([1.0, 2.0, 3.0])
      b = et([1.0, 2.0, 99.0])
      assert_equiv(&Nx.all_close/2, [a, b])
    end

    test "all_close with custom rtol/atol matches" do
      a = et([1.0, 2.0, 3.0])
      b = et([1.1, 2.0, 3.0])
      assert_equiv(fn a, b -> Nx.all_close(a, b, atol: 0.2) end, [a, b])
      assert_equiv(fn a, b -> Nx.all_close(a, b, atol: 0.05) end, [a, b])
    end

    test "all_close with equal_nan: true matches (NaN compares equal to NaN)" do
      a = et([1.0, :nan, 3.0])
      b = et([1.0, :nan, 3.0])
      assert_equiv(fn a, b -> Nx.all_close(a, b, equal_nan: true) end, [a, b])
      assert_equiv(fn a, b -> Nx.all_close(a, b, equal_nan: false) end, [a, b])
    end

    # Nx.phase (Nx.Block.Phase) := atan2(imag(t), real(t)). Backend
    # falls through to the composed expansion (no fused kernel); the IR
    # lowers the same expansion via the TopK-style param-seeding, so the
    # result is bit-identical to the evaluator. Exercise the four
    # quadrants plus the real axis (phase == 0 for real, ±pi for
    # negative real) and the imaginary axis (±pi/2).
    test "phase matches the evaluator across the four complex quadrants" do
      x =
        et([
          Complex.new(1.0, 1.0),
          Complex.new(-1.0, 1.0),
          Complex.new(-1.0, -1.0),
          Complex.new(1.0, -1.0),
          Complex.new(2.0, 0.0),
          Complex.new(-2.0, 0.0),
          Complex.new(0.0, 3.0),
          Complex.new(0.0, -3.0)
        ])

      out = assert_equiv(&Nx.phase/1, [x])
      assert out.type == {:f, 32}
    end
  end

  describe "pad / triangular_solve" do
    # pad with constant value matches Emily.Backend.pad/4 (mx::pad with
    # mode "constant", no interior dilation). Cover symmetric and
    # asymmetric padding on both 1-D and 2-D inputs.
    test "constant-pad on 1-D and 2-D inputs matches the evaluator" do
      x = et([1.0, 2.0, 3.0])
      assert_equiv(fn t -> Nx.pad(t, 0.0, [{1, 1, 0}]) end, [x])
      assert_equiv(fn t -> Nx.pad(t, -1.0, [{2, 0, 0}]) end, [x])

      y = et([[1.0, 2.0], [3.0, 4.0]])
      assert_equiv(fn t -> Nx.pad(t, 0.0, [{1, 1, 0}, {2, 2, 0}]) end, [y])
      assert_equiv(fn t -> Nx.pad(t, 99.0, [{0, 1, 0}, {1, 0, 0}]) end, [y])
    end

    # Interior padding (interior > 0) has no MLX primitive — both the
    # eager Backend and the IR raise on it. The IR's raise must surface
    # cleanly (the graceful `:eval` fallback path is what catches it in
    # production; here we assert the underlying behaviour).
    test "pad with interior > 0 raises in the IR (no fallback)" do
      x = et([1.0, 2.0, 3.0])

      assert_raise ArgumentError, ~r/interior > 0/, fn ->
        run(fn t -> Nx.pad(t, 0.0, [{0, 0, 1}]) end, [x],
          compiler: Emily.Compiler,
          native: true,
          native_fallback: :raise
        )
      end
    end

    # triangular_solve over the four `transform_a` × `left_side`
    # combinations. The IR composes each case with transposes around the
    # bare `linalg_solve_triangular` opcode (operands [a, b]; iattrs
    # [[upper]]) — exactly the sequence Emily.Backend.triangular_solve/4
    # uses. mx::linalg::solve_triangular is CPU-only; the C++ dispatcher
    # overrides the stream per call, mirroring c_src/ops/linalg.cpp.
    test "lower-triangular A x = b (default opts) matches the evaluator" do
      a = et([[2.0, 0.0], [3.0, 1.0]])
      b = et([4.0, 9.0])
      assert_equiv(&Nx.LinAlg.triangular_solve/2, [a, b])
    end

    test "upper-triangular A x = b matches" do
      a = et([[2.0, 1.0], [0.0, 3.0]])
      b = et([5.0, 6.0])
      assert_equiv(fn a, b -> Nx.LinAlg.triangular_solve(a, b, lower: false) end, [a, b])
    end

    test "transposed A (transform_a: :transpose, left_side: true) matches" do
      a = et([[2.0, 0.0], [3.0, 1.0]])
      b = et([4.0, 9.0])

      assert_equiv(
        fn a, b -> Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose) end,
        [a, b]
      )
    end

    test "right-side solve x A = b (left_side: false) matches" do
      a = et([[2.0, 0.0], [3.0, 1.0]])
      b = et([[4.0, 9.0], [1.0, 2.0]])

      assert_equiv(
        fn a, b -> Nx.LinAlg.triangular_solve(a, b, left_side: false) end,
        [a, b]
      )
    end

    test "transposed right-side solve (transform_a: :transpose, left_side: false) matches" do
      a = et([[2.0, 0.0], [3.0, 1.0]])
      b = et([[4.0, 9.0], [1.0, 2.0]])

      assert_equiv(
        fn a, b ->
          Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose, left_side: false)
        end,
        [a, b]
      )
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

    # SDPA with attention sinks (gpt-oss). Backend dispatches both the
    # plain sinks variant and the masked sinks variant through the same
    # mx::fast::scaled_dot_product_attention kernel (with the `sinks` arg
    # populated). The native IR routes through new fast_sdpa_sinks /
    # fast_sdpa_mask_sinks opcodes that call the same kernel — so the two
    # paths are bit-identical to the evaluator.
    test "sdpa with sinks (non-causal) matches the fused kernel" do
      shape = {1, 2, 4, 8}
      q = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      k = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      v = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(32.0)
      # Sinks is a per-head 1-D tensor (one extra logit per head).
      sinks = et([0.5, -0.5])

      assert_equiv(
        fn q, k, v, s -> Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s) end,
        [q, k, v, sinks]
      )
    end

    test "sdpa with sinks (causal) matches the fused kernel" do
      shape = {1, 2, 4, 8}
      q = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      k = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      v = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(32.0)
      sinks = et([0.0, 1.0])

      assert_equiv(
        fn q, k, v, s ->
          Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s, causal: true)
        end,
        [q, k, v, sinks]
      )
    end

    test "sdpa with an additive mask AND sinks matches the fused kernel" do
      shape = {1, 2, 4, 8}
      q = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(64.0)
      k = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(48.0)
      v = Nx.iota(shape, type: :f32, backend: Emily.Backend) |> Nx.divide(32.0)
      mask = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {1, 1, 4, 4})
      sinks = et([-1.0, 1.0])

      assert_equiv(
        fn q, k, v, m, s ->
          Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m, sinks: s)
        end,
        [q, k, v, mask, sinks]
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

  describe "scatter (indexed_put / indexed_add)" do
    test "indexed_put / indexed_add into a 2-D target match the evaluator" do
      target = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      # Two distinct cells in the {2,3} grid -> order-independent.
      idx = Nx.tensor([[0, 0], [1, 2]], type: :s64, backend: Emily.Backend)
      upd = et([10.0, 20.0])

      assert_equiv(fn t, i, u -> Nx.indexed_put(t, i, u) end, [target, idx, upd])
      assert_equiv(fn t, i, u -> Nx.indexed_add(t, i, u) end, [target, idx, upd])
    end

    test "indexed_add accumulates duplicate indices (same MLX kernel both paths)" do
      target = et([0.0, 0.0, 0.0, 0.0])
      idx = Nx.tensor([[1], [1], [3]], type: :s64, backend: Emily.Backend)
      upd = et([5.0, 7.0, 2.0])
      assert_equiv(fn t, i, u -> Nx.indexed_add(t, i, u) end, [target, idx, upd])
    end

    test "indexed_put on a partial axis set (axes: [0], whole-row writes) matches" do
      target = et([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      idx = Nx.tensor([[0], [2]], type: :s64, backend: Emily.Backend)
      upd = et([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
      assert_equiv(fn t, i, u -> Nx.indexed_put(t, i, u, axes: [0]) end, [target, idx, upd])
    end
  end

  describe "window reductions (pooling forward)" do
    test "2x2 maxpool / sumpool / minpool (CNN-shaped) match the Evaluator" do
      # {batch, channels, h, w}; pool only the spatial axes, stride 2.
      x = Nx.iota({1, 2, 4, 4}, type: :f32, backend: Emily.Backend) |> Nx.divide(16.0)

      assert_equiv(fn t -> Nx.window_max(t, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) end, [x])
      assert_equiv(fn t -> Nx.window_sum(t, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) end, [x])
      assert_equiv(fn t -> Nx.window_min(t, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) end, [x])
    end

    test "maxpool with padding (boundary identity) matches" do
      x = Nx.iota({1, 1, 5, 5}, type: :f32, backend: Emily.Backend) |> Nx.divide(25.0)

      assert_equiv(
        fn t ->
          Nx.window_max(t, {1, 1, 3, 3},
            strides: [1, 1, 2, 2],
            padding: [{0, 0}, {0, 0}, {1, 1}, {1, 1}]
          )
        end,
        [x]
      )
    end

    test "window_product matches" do
      x =
        Nx.iota({1, 8}, type: :f32, backend: Emily.Backend) |> Nx.divide(8.0) |> Nx.add(1.0)

      assert_equiv(fn t -> Nx.window_product(t, {1, 2}, strides: [1, 1]) end, [x])
    end

    test "dilated windows match (issue #175)" do
      # Tiny input + dilation 2 on the inner axis: the sliding-window view
      # over-reads its aliased buffer unless materialised before the reduce.
      # Pre-fix the native path returned garbage here while the Evaluator
      # returned a deterministic 0.0, so the two diverged.
      x =
        Nx.iota({1, 8}, type: :f32, backend: Emily.Backend) |> Nx.divide(8.0) |> Nx.add(1.0)

      assert_equiv(fn t -> Nx.window_max(t, {1, 3}, window_dilations: [1, 2]) end, [x])
      assert_equiv(fn t -> Nx.window_sum(t, {1, 3}, window_dilations: [1, 2]) end, [x])
      assert_equiv(fn t -> Nx.window_min(t, {1, 3}, window_dilations: [1, 2]) end, [x])
    end
  end

  describe "window scatter (pooling backward)" do
    # The MaxPool/MinPool backward: scatter the upstream gradient `source`
    # (one per pooled window) into the argmax/argmin position of each window.
    test "window_scatter_max / window_scatter_min match the Evaluator" do
      t = Nx.iota({1, 1, 4, 4}, type: :f32, backend: Emily.Backend) |> Nx.divide(16.0)
      source = Nx.iota({1, 1, 2, 2}, type: :f32, backend: Emily.Backend) |> Nx.add(1.0)

      assert_equiv(
        fn t, src ->
          Nx.window_scatter_max(t, src, 0.0, {1, 1, 2, 2}, strides: [1, 1, 2, 2])
        end,
        [t, source]
      )

      assert_equiv(
        fn t, src ->
          Nx.window_scatter_min(t, src, 0.0, {1, 1, 2, 2}, strides: [1, 1, 2, 2])
        end,
        [t, source]
      )
    end

    test "grad(window_max) lowers (the maxpool backward path) and matches" do
      x = Nx.iota({1, 1, 4, 4}, type: :f32, backend: Emily.Backend) |> Nx.divide(16.0)

      assert_equiv(
        fn t ->
          Nx.Defn.grad(t, fn t ->
            t |> Nx.window_max({1, 1, 2, 2}, strides: [1, 1, 2, 2]) |> Nx.sum()
          end)
        end,
        [x]
      )
    end
  end

  describe "top_k (multi-output block)" do
    test "values and indices both match the evaluator" do
      x = et([[3.0, 1.0, 4.0, 1.5, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]])

      {nv, ni} = run(fn t -> Nx.top_k(t, k: 3) end, [x], @native)
      {ev, ei} = run(fn t -> Nx.top_k(t, k: 3) end, [x], @eval)

      assert %Emily.Backend{} = nv.data
      assert Nx.to_binary(nv) == Nx.to_binary(ev)
      assert Nx.to_binary(ni) == Nx.to_binary(ei)
      assert ni.type == {:s, 32}
    end

    test "top_k feeding a downstream op lowers (both leaves projected)" do
      x = et([[3.0, 1.0, 4.0, 1.5, 5.0]])

      assert_equiv(
        fn t ->
          {v, i} = Nx.top_k(t, k: 2)
          Nx.add(v, Nx.as_type(i, :f32))
        end,
        [x]
      )
    end

    test "top_k binds its block param to the real arg, not outer {:input, 0} (regression)" do
      # Mirrors the ModernBERT failure: outer input 0 is lower-rank than the
      # top_k input. The `Nx.Block.TopK` expansion is built against fresh
      # parameters, so an unbound parameter falls through to `{:input, 0}` —
      # here the 1-D tensor — and argsort hits "axis 1 on a 1-D array".
      one_d = et([0.0, 0.0])
      scores = et([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])

      {nv, ni} = run(fn _i, s -> Nx.top_k(s, k: 2) end, [one_d, scores], @native)
      {ev, ei} = run(fn _i, s -> Nx.top_k(s, k: 2) end, [one_d, scores], @eval)

      assert Nx.to_binary(nv) == Nx.to_binary(ev)
      assert Nx.to_binary(ni) == Nx.to_binary(ei)
    end
  end

  describe "fft family (signal transforms)" do
    test "1-D fft / ifft on the trailing axis match the evaluator" do
      x = et([1.0, 2.0, 3.0, 4.0])
      assert_equiv(fn t -> Nx.fft(t) end, [x])
      assert_equiv(fn t -> Nx.ifft(t) end, [x])
    end

    test "fft with explicit length (zero-pad / truncate) matches" do
      x = et([1.0, 2.0, 3.0, 4.0, 5.0])
      assert_equiv(fn t -> Nx.fft(t, length: 8) end, [x])
      assert_equiv(fn t -> Nx.fft(t, length: 4) end, [x])
    end

    test "batched 1-D fft transforms the last axis only" do
      x = et([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      assert_equiv(fn t -> Nx.fft(t) end, [x])
    end

    test "ifft(fft(x)) round-trips through a complex intermediate" do
      x = et([1.0, -2.0, 3.0, -4.0])
      out = assert_equiv(fn t -> Nx.ifft(Nx.fft(t)) end, [x])
      assert out.type == {:c, 64}
    end

    test "2-D fft2 / ifft2 (Nx.Block) match" do
      x = et([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      assert_equiv(fn t -> Nx.fft2(t) end, [x])
      assert_equiv(fn t -> Nx.ifft2(t) end, [x])
    end

    test "rfft (real -> half spectrum) and irfft (back to real) match" do
      x = et([1.0, 2.0, 3.0, 4.0])
      out = assert_equiv(fn t -> Nx.rfft(t) end, [x])
      assert out.type == {:c, 64}
      # irfft takes the complex half-spectrum back to a real signal.
      assert_equiv(fn t -> Nx.irfft(Nx.rfft(t)) end, [x])
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
