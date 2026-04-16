defmodule Emily.BackendTest do
  @moduledoc """
  Property-based oracle tests for `Emily.Backend`. Each property
  generates a random tensor (or pair), runs an op under Emily, and
  asserts the result matches `Nx.BinaryBackend` within a dtype-aware
  tolerance.

  Rank is capped at 4, dimensions at 6, values in [-10, 10] for floats
  and small integer ranges. This keeps property runs fast while still
  exercising broadcasting, reductions, and the u8↔pred coercion.
  """

  use ExUnit.Case, async: true
  use ExUnitProperties

  import Emily.BackendGenerators

  @max_runs 25

  # ---------------- Creation ----------------

  describe "creation" do
    property "iota matches BinaryBackend" do
      check all(
              shape <- non_scalar_shape(),
              axis <- StreamData.one_of([StreamData.constant(nil), axis_of(shape)]),
              max_runs: @max_runs
            ) do
        emily = Nx.iota(shape, backend: Emily.Backend, axis: axis)
        ref = Nx.iota(shape, backend: Nx.BinaryBackend, axis: axis)
        assert_close(emily, ref)
      end
    end

    property "eye matches BinaryBackend (2D)" do
      check all(
              n <- StreamData.integer(1..5),
              m <- StreamData.integer(1..5),
              max_runs: @max_runs
            ) do
        emily = Nx.eye({n, m}, backend: Emily.Backend, type: {:f, 32})
        ref = Nx.eye({n, m}, backend: Nx.BinaryBackend, type: {:f, 32})
        assert_close(emily, ref)
      end
    end

    test "constant broadcasts a scalar" do
      emily = Nx.broadcast(Nx.tensor(3.5, backend: Emily.Backend), {2, 3})
      ref = Nx.broadcast(Nx.tensor(3.5, backend: Nx.BinaryBackend), {2, 3})
      assert_close(emily, ref)
    end
  end

  # ---------------- Cast ----------------

  describe "as_type" do
    property "f32 -> s32 truncates toward zero" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.as_type(to_emily(a), {:s, 32})
        ref = Nx.as_type(a, {:s, 32})
        assert_close(emily, ref)
      end
    end

    property "s32 -> f32 is exact for small ints" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:s, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.as_type(to_emily(a), {:f, 32})
        ref = Nx.as_type(a, {:f, 32})
        assert_close(emily, ref)
      end
    end
  end

  # ---------------- Unary math ----------------

  @unary_math [:exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh, :sqrt, :rsqrt]
  # :round is excluded — MLX uses banker's rounding (half-to-even) while
  # BinaryBackend uses half-away-from-zero; see Emily.NativeTest for a
  # targeted test covering MLX's behaviour.
  @unary_basic [:negate, :abs, :ceil, :floor, :sign]
  @unary_logical [:logical_not, :is_nan, :is_infinity]

  describe "unary math" do
    for op <- @unary_math do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:f, 32}),
                max_runs: @max_runs
              ) do
          a = sanitise_unary_arg(op, a)
          emily = apply(Nx, op, [to_emily(a)])
          ref = apply(Nx, op, [a])
          assert_close(emily, ref)
        end
      end
    end
  end

  describe "unary basic" do
    for op <- @unary_basic do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:f, 32}),
                max_runs: @max_runs
              ) do
          emily = apply(Nx, op, [to_emily(a)])
          ref = apply(Nx, op, [a])
          assert_close(emily, ref)
        end
      end
    end
  end

  describe "unary logical" do
    for op <- @unary_logical do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:f, 32}),
                max_runs: @max_runs
              ) do
          emily = apply(Nx, op, [to_emily(a)])
          ref = apply(Nx, op, [a])
          assert_close(emily, ref)
        end
      end
    end
  end

  # ---------------- Binary elementwise ----------------

  @float_binary [:add, :subtract, :multiply, :divide, :min, :max, :pow]
  @bool_binary [:equal, :not_equal, :less, :less_equal, :greater, :greater_equal]
  @logical_binary [:logical_and, :logical_or, :logical_xor]
  @int_binary [:bitwise_and, :bitwise_or, :bitwise_xor]

  describe "float binary ops" do
    for op <- @float_binary do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:f, 32}),
                b <- tensor(shape, {:f, 32}),
                max_runs: @max_runs
              ) do
          {a, b} = sanitise_binary_args(op, a, b)
          emily = apply(Nx, op, [to_emily(a), to_emily(b)])
          ref = apply(Nx, op, [a, b])
          assert_close(emily, ref, tol: 1.0e-3)
        end
      end
    end
  end

  describe "boolean-producing binary ops" do
    for op <- @bool_binary do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:s, 32}),
                b <- tensor(shape, {:s, 32}),
                max_runs: @max_runs
              ) do
          emily = apply(Nx, op, [to_emily(a), to_emily(b)])
          ref = apply(Nx, op, [a, b])
          assert_nx_eq(emily, ref)
        end
      end
    end
  end

  describe "logical binary ops" do
    for op <- @logical_binary do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:u, 8}),
                b <- tensor(shape, {:u, 8}),
                max_runs: @max_runs
              ) do
          emily = apply(Nx, op, [to_emily(a), to_emily(b)])
          ref = apply(Nx, op, [a, b])
          assert_nx_eq(emily, ref)
        end
      end
    end
  end

  describe "int bitwise ops" do
    for op <- @int_binary do
      property "#{op} matches BinaryBackend" do
        op = unquote(op)

        check all(
                shape <- non_scalar_shape(),
                a <- tensor(shape, {:s, 32}),
                b <- tensor(shape, {:s, 32}),
                max_runs: @max_runs
              ) do
          emily = apply(Nx, op, [to_emily(a), to_emily(b)])
          ref = apply(Nx, op, [a, b])
          assert_nx_eq(emily, ref)
        end
      end
    end
  end

  # ---------------- Shape ----------------

  describe "shape ops" do
    property "reshape round-trips" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        size = Nx.size(shape)
        new_shape = {size}
        emily = Nx.reshape(to_emily(a), new_shape)
        ref = Nx.reshape(a, new_shape)
        assert_close(emily, ref)
      end
    end

    property "transpose reverses axes by default" do
      check all(
              rank <- StreamData.integer(1..4),
              shape <- shape(rank),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.transpose(to_emily(a))
        ref = Nx.transpose(a)
        assert_close(emily, ref)
      end
    end

    property "concatenate along axis 0" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              b <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.concatenate([to_emily(a), to_emily(b)], axis: 0)
        ref = Nx.concatenate([a, b], axis: 0)
        assert_close(emily, ref)
      end
    end

    property "stack along axis 0" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              b <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.stack([to_emily(a), to_emily(b)], axis: 0)
        ref = Nx.stack([a, b], axis: 0)
        assert_close(emily, ref)
      end
    end

    test "broadcast along prefix" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      emily = Nx.broadcast(to_emily(a), {2, 3})
      ref = Nx.broadcast(a, {2, 3})
      assert_close(emily, ref)
    end

    test "pad edges" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      emily = Nx.pad(to_emily(a), 0.0, [{1, 1, 0}, {2, 0, 0}])
      ref = Nx.pad(a, 0.0, [{1, 1, 0}, {2, 0, 0}])
      assert_close(emily, ref)
    end

    test "reverse along all axes" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      emily = Nx.reverse(to_emily(a))
      ref = Nx.reverse(a)
      assert_close(emily, ref)
    end
  end

  # ---------------- Indexing ----------------

  describe "indexing" do
    property "slice matches BinaryBackend" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        starts = shape |> Tuple.to_list() |> Enum.map(fn d -> div(d, 2) end)
        lengths = shape |> Tuple.to_list() |> Enum.map(fn d -> max(1, d - div(d, 2)) end)
        strides = List.duplicate(1, tuple_size(shape))
        emily = Nx.slice(to_emily(a), starts, lengths, strides: strides)
        ref = Nx.slice(a, starts, lengths, strides: strides)
        assert_close(emily, ref)
      end
    end

    test "select returns the correct branch" do
      pred = Nx.tensor([[1, 0], [0, 1]], type: {:u, 8})
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])
      emily = Nx.select(to_emily(pred), to_emily(a), to_emily(b))
      ref = Nx.select(pred, a, b)
      assert_close(emily, ref)
    end

    test "clip bounds values" do
      a = Nx.tensor([-2.0, -0.5, 0.5, 2.0])
      emily = Nx.clip(to_emily(a), -1.0, 1.0)
      ref = Nx.clip(a, -1.0, 1.0)
      assert_close(emily, ref)
    end

    test "take along axis 0" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      idx = Nx.tensor([2, 0, 1], type: {:s, 32})
      emily = Nx.take(to_emily(a), to_emily(idx), axis: 0)
      ref = Nx.take(a, idx, axis: 0)
      assert_close(emily, ref)
    end
  end

  # ---------------- Reductions ----------------

  describe "reductions" do
    property "sum over all axes matches BinaryBackend" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.sum(to_emily(a))
        ref = Nx.sum(a)
        assert_close(emily, ref, tol: 1.0e-3)
      end
    end

    property "sum along a single axis matches BinaryBackend" do
      check all(
              shape <- non_scalar_shape(),
              axis <- axis_of(shape),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.sum(to_emily(a), axes: [axis])
        ref = Nx.sum(a, axes: [axis])
        assert_close(emily, ref, tol: 1.0e-3)
      end
    end

    property "product along a single axis matches BinaryBackend" do
      check all(
              shape <- non_scalar_shape(),
              axis <- axis_of(shape),
              a <- tensor(shape, {:s, 32}),
              max_runs: @max_runs
            ) do
        # Keep values small to avoid int32 overflow.
        a = Nx.remainder(a, 3)
        emily = Nx.product(to_emily(a), axes: [axis])
        ref = Nx.product(a, axes: [axis])
        assert_nx_eq(emily, ref)
      end
    end

    property "reduce_max along a single axis" do
      check all(
              shape <- non_scalar_shape(),
              axis <- axis_of(shape),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.reduce_max(to_emily(a), axes: [axis])
        ref = Nx.reduce_max(a, axes: [axis])
        assert_close(emily, ref)
      end
    end

    property "argmax along a single axis" do
      check all(
              shape <- non_scalar_shape(),
              axis <- axis_of(shape),
              max_runs: @max_runs
            ) do
        # Use distinct values to avoid tie-break divergence between MLX
        # and BinaryBackend.
        a = Nx.iota(shape, type: {:f, 32}, axis: axis)
        # Permute so argmax is non-trivial.
        flipped = Nx.reverse(a, axes: [axis])
        emily_a = to_emily(flipped)

        emily = Nx.argmax(emily_a, axis: axis) |> Nx.as_type({:s, 64})
        ref = Nx.argmax(flipped, axis: axis) |> Nx.as_type({:s, 64})
        assert_nx_eq(emily, ref)
      end
    end

    property "cumulative_sum" do
      check all(
              shape <- non_scalar_shape(),
              axis <- axis_of(shape),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.cumulative_sum(to_emily(a), axis: axis)
        ref = Nx.cumulative_sum(a, axis: axis)
        assert_close(emily, ref, tol: 1.0e-3)
      end
    end
  end

  # ---------------- Dot ----------------

  describe "dot" do
    test "matrix-matrix non-batched" do
      a = Nx.iota({3, 4}, type: {:f, 32})
      b = Nx.iota({4, 5}, type: {:f, 32})
      emily = Nx.dot(to_emily(a), to_emily(b))
      ref = Nx.dot(a, b)
      assert_close(emily, ref)
    end

    test "vector-vector" do
      a = Nx.iota({5}, type: {:f, 32})
      b = Nx.iota({5}, type: {:f, 32})
      emily = Nx.dot(to_emily(a), to_emily(b))
      ref = Nx.dot(a, b)
      assert_close(emily, ref)
    end

    # Batched dot is the transformer-attention hot path. We test the
    # permute + reshape + matmul translation directly here rather than
    # relying on the DistilBERT conformance run to catch misbehaviour.
    property "1 batch axis: (B, M, K) x (B, K, N)" do
      check all(
              b <- StreamData.integer(1..4),
              m <- StreamData.integer(1..4),
              k <- StreamData.integer(1..4),
              n <- StreamData.integer(1..4),
              a <- tensor({b, m, k}, {:f, 32}),
              c <- tensor({b, k, n}, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.dot(to_emily(a), [2], [0], to_emily(c), [1], [0])
        ref = Nx.dot(a, [2], [0], c, [1], [0])
        assert_close(emily, ref, tol: 1.0e-3)
      end
    end

    # Models the DistilBERT attention shape post-head-split:
    # query/key {batch, heads, seq, head_dim} with batch=[0,1],
    # contract=[3], free=[2] → weights {batch, heads, seq_q, seq_k}.
    property "2 batch axes: attention weights" do
      check all(
              bs <- StreamData.integer(1..2),
              heads <- StreamData.integer(1..3),
              seq <- StreamData.integer(2..4),
              head_dim <- StreamData.integer(2..4),
              q <- tensor({bs, heads, seq, head_dim}, {:f, 32}),
              k <- tensor({bs, heads, seq, head_dim}, {:f, 32}),
              max_runs: @max_runs
            ) do
        emily = Nx.dot(to_emily(q), [3], [0, 1], to_emily(k), [3], [0, 1])
        ref = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
        assert_close(emily, ref, tol: 1.0e-3)
      end
    end

    # Scalar-free output (contract all non-batch dims on both sides).
    test "batched vector-vector" do
      a = Nx.iota({3, 5}, type: {:f, 32})
      b = Nx.iota({3, 5}, type: {:f, 32})
      emily = Nx.dot(to_emily(a), [1], [0], to_emily(b), [1], [0])
      ref = Nx.dot(a, [1], [0], b, [1], [0])
      assert_close(emily, ref, tol: 1.0e-3)
    end

    test "batched matmul with multiple free axes on each side" do
      # a: {B, M1, M2, K}, b: {B, K, N1, N2} → {B, M1, M2, N1, N2}
      a = Nx.iota({2, 2, 3, 4}, type: {:f, 32}) |> Nx.divide(100.0)
      b = Nx.iota({2, 4, 2, 3}, type: {:f, 32}) |> Nx.divide(100.0)
      emily = Nx.dot(to_emily(a), [3], [0], to_emily(b), [1], [0])
      ref = Nx.dot(a, [3], [0], b, [1], [0])
      assert_close(emily, ref, tol: 1.0e-3)
    end
  end

  # ---------------- Sort ----------------

  describe "sort" do
    property "sort along last axis asc" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        axis = tuple_size(shape) - 1
        emily = Nx.sort(to_emily(a), axis: axis)
        ref = Nx.sort(a, axis: axis)
        assert_close(emily, ref)
      end
    end

    property "sort along last axis desc" do
      check all(
              shape <- non_scalar_shape(),
              a <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        axis = tuple_size(shape) - 1
        emily = Nx.sort(to_emily(a), axis: axis, direction: :desc)
        ref = Nx.sort(a, axis: axis, direction: :desc)
        assert_close(emily, ref)
      end
    end
  end

  # ---------------- Indexing: gather / indexed_add / indexed_put ----------------
  #
  # Targeted tests rather than properties: valid gather/scatter inputs
  # need coordinated construction (indices bounded per axis, updates
  # shape derived from target + axes) that StreamData makes awkward.
  # Each case exercises a distinct native-translation path.

  describe "gather" do
    test "multi-axis scalar picks (axes = [0, 1])" do
      a = Nx.iota({3, 4}, type: {:f, 32}, backend: Emily.Backend) |> Nx.add(1.0)
      idx = Nx.tensor([[0, 1], [2, 3], [1, 0]], backend: Emily.Backend)

      emily = Nx.gather(a, idx, axes: [0, 1])
      ref = Nx.gather(bin(a), bin(idx), axes: [0, 1])
      assert_nx_eq(emily, ref)
    end

    test "partial-axis gather keeps remaining dim (axes = [0])" do
      a = Nx.iota({3, 4}, type: {:f, 32}, backend: Emily.Backend) |> Nx.add(1.0)
      idx = Nx.tensor([[2], [0]], backend: Emily.Backend)

      emily = Nx.gather(a, idx, axes: [0])
      ref = Nx.gather(bin(a), bin(idx), axes: [0])
      assert_nx_eq(emily, ref)
    end

    test "multi-dim batch in indices" do
      a = Nx.iota({3, 4}, type: {:f, 32}, backend: Emily.Backend) |> Nx.add(1.0)
      idx = Nx.tensor([[[0, 1], [2, 3]], [[1, 0], [2, 2]]], backend: Emily.Backend)

      emily = Nx.gather(a, idx, axes: [0, 1])
      ref = Nx.gather(bin(a), bin(idx), axes: [0, 1])
      assert_nx_eq(emily, ref)
    end
  end

  describe "indexed_add" do
    test "scalar writes across all axes" do
      a = Nx.broadcast(0.0, {2, 3}) |> Nx.backend_transfer(Emily.Backend)
      idx = Nx.tensor([[0, 1], [1, 2], [0, 0]], backend: Emily.Backend)
      upd = Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)

      emily = Nx.indexed_add(a, idx, upd)
      ref = Nx.indexed_add(bin(a), bin(idx), bin(upd))
      assert_nx_eq(emily, ref)
    end

    test "partial-axis slice writes on a {B, L, D} target" do
      a = Nx.broadcast(0.0, {2, 3, 4}) |> Nx.backend_transfer(Emily.Backend)
      idx = Nx.tensor([[0, 1], [1, 2]], backend: Emily.Backend)
      # updates shape: {2, 4} — one D-vector per write.
      upd =
        Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], backend: Emily.Backend)

      emily = Nx.indexed_add(a, idx, upd, axes: [0, 1])
      ref = Nx.indexed_add(bin(a), bin(idx), bin(upd), axes: [0, 1])
      assert_nx_eq(emily, ref)
    end

    test "duplicate indices accumulate (commutative)" do
      a = Nx.tensor([10.0, 20.0, 30.0], backend: Emily.Backend)
      idx = Nx.tensor([[0], [0], [2]], backend: Emily.Backend)
      upd = Nx.tensor([1.0, 1.0, 5.0], backend: Emily.Backend)

      emily = Nx.indexed_add(a, idx, upd)
      ref = Nx.indexed_add(bin(a), bin(idx), bin(upd))
      assert_nx_eq(emily, ref)
    end
  end

  describe "indexed_put" do
    test "scalar writes across all axes (unique indices)" do
      a = Nx.broadcast(0.0, {2, 3}) |> Nx.backend_transfer(Emily.Backend)
      idx = Nx.tensor([[0, 1], [1, 2], [1, 0]], backend: Emily.Backend)
      upd = Nx.tensor([7.0, 8.0, 9.0], backend: Emily.Backend)

      emily = Nx.indexed_put(a, idx, upd)
      ref = Nx.indexed_put(bin(a), bin(idx), bin(upd))
      assert_nx_eq(emily, ref)
    end

    test "partial-axis slice writes (unique indices)" do
      a = Nx.broadcast(0.0, {2, 3, 4}) |> Nx.backend_transfer(Emily.Backend)
      idx = Nx.tensor([[0, 1], [1, 2]], backend: Emily.Backend)

      upd =
        Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], backend: Emily.Backend)

      emily = Nx.indexed_put(a, idx, upd, axes: [0, 1])
      ref = Nx.indexed_put(bin(a), bin(idx), bin(upd), axes: [0, 1])
      assert_nx_eq(emily, ref)
    end
  end

  # ---------------- Helpers ----------------

  # Short alias for backend-transferring to BinaryBackend inside inline
  # oracle expressions.
  defp bin(t), do: Nx.backend_transfer(t, Nx.BinaryBackend)

  defp axis_of(shape) do
    StreamData.integer(0..(tuple_size(shape) - 1))
  end

  defp assert_nx_eq(%Nx.Tensor{} = a, %Nx.Tensor{} = b) do
    assert_close(a, b, tol: 0)
  end

  # Adjust args to sidestep BinaryBackend edge cases (division by zero,
  # 0^negative, negative^fraction) that aren't the subject of the test.
  defp sanitise_binary_args(:divide, a, b), do: {a, Nx.add(Nx.abs(b), 0.1)}

  defp sanitise_binary_args(:pow, a, b) do
    base = Nx.add(Nx.abs(a), 0.5)
    exp = Nx.multiply(Nx.abs(b), 0.3)
    {base, exp}
  end

  defp sanitise_binary_args(_op, a, b), do: {a, b}

  # log/sqrt/rsqrt need strictly positive inputs to avoid NaN mismatches
  # with BinaryBackend (which returns NaN, while Nx ≠ Emily on NaN eq).
  defp sanitise_unary_arg(op, a) when op in [:log, :sqrt, :rsqrt] do
    Nx.add(Nx.abs(a), 0.5)
  end

  defp sanitise_unary_arg(_op, a), do: a

  # ---------------- Linalg ----------------

  describe "linalg" do
    property "lu: P * L * U ≈ A for random well-conditioned matrices" do
      check all(a <- square_matrix(), max_runs: @max_runs) do
        # Diagonal dominance ensures non-singularity.
        n = elem(Nx.shape(a), 0)
        a_safe = Nx.add(a, Nx.multiply(Nx.eye(n, backend: Nx.BinaryBackend), n * 10))
        emily_a = to_emily(a_safe)
        {p, l, u} = Nx.LinAlg.lu(emily_a)
        reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
        assert_close(reconstructed, emily_a, tol: 1.0e-3)
      end
    end

    property "svd: singular values match BinaryBackend" do
      check all(a <- square_matrix(), max_runs: @max_runs) do
        emily_a = to_emily(a)
        {_u, emily_s, _vt} = Nx.LinAlg.svd(emily_a)
        {_u, ref_s, _vt} = Nx.LinAlg.svd(a)

        # Compare sorted singular values (sign/ordering of U/Vt is
        # ambiguous, but singular values themselves must match).
        assert_close(Nx.sort(emily_s), Nx.sort(ref_s), tol: 1.0e-3)
      end
    end

    property "qr: Q * R ≈ A for random well-conditioned matrices" do
      check all(a <- square_matrix(), max_runs: @max_runs) do
        n = elem(Nx.shape(a), 0)
        a_safe = Nx.add(a, Nx.multiply(Nx.eye(n, backend: Nx.BinaryBackend), n * 10))
        emily_a = to_emily(a_safe)
        {q, r} = Nx.LinAlg.qr(emily_a, mode: :reduced)
        reconstructed = Nx.dot(q, r)
        assert_close(reconstructed, emily_a, tol: 1.0e-3)
      end
    end

    property "cholesky: L * L^T ≈ A for random SPD matrices" do
      check all(a <- spd_matrix(), max_runs: @max_runs) do
        emily_a = to_emily(a)
        l = Nx.LinAlg.cholesky(emily_a)
        reconstructed = Nx.dot(l, Nx.transpose(l))
        assert_close(reconstructed, emily_a, tol: 1.0e-3)
      end
    end

    property "eigh: eigenvalues match BinaryBackend for symmetric matrices" do
      check all(a <- symmetric_matrix(), max_runs: @max_runs) do
        emily_a = to_emily(a)
        {emily_vals, _emily_vecs} = Nx.LinAlg.eigh(emily_a)
        {ref_vals, _ref_vecs} = Nx.LinAlg.eigh(a)

        # Sort eigenvalues before comparing (ordering may differ)
        assert_close(Nx.sort(emily_vals), Nx.sort(ref_vals), tol: 1.0e-3)
      end
    end

    property "solve: Ax = b matches BinaryBackend" do
      check all(a <- square_matrix(), max_runs: @max_runs) do
        n = elem(Nx.shape(a), 0)
        b = Nx.iota({n}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.add(1.0)

        # Add diagonal dominance to ensure non-singularity
        a_safe = Nx.add(a, Nx.multiply(Nx.eye(n, backend: Nx.BinaryBackend), n * 10))

        emily_x = Nx.LinAlg.solve(to_emily(a_safe), to_emily(b))
        ref_x = Nx.LinAlg.solve(a_safe, b)
        assert_close(emily_x, ref_x, tol: 1.0e-3)
      end
    end

    property "triangular_solve matches BinaryBackend for lower-triangular" do
      check all(l <- lower_triangular_matrix(), max_runs: @max_runs) do
        n = elem(Nx.shape(l), 0)
        b = Nx.iota({n}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.add(1.0)

        emily_x = Nx.LinAlg.triangular_solve(to_emily(l), to_emily(b))
        ref_x = Nx.LinAlg.triangular_solve(l, b)
        assert_close(emily_x, ref_x, tol: 1.0e-3)
      end
    end
  end
end
