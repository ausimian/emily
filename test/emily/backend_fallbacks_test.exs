defmodule Emily.Backend.FallbacksTest do
  @moduledoc """
  Tests that exercise every `via_binary` fallback path in
  `Emily.Backend`. These are backend callbacks with no single MLX
  primitive — we transfer inputs to `Nx.BinaryBackend`, run the
  reference op there, and transfer the result back.

  Because the fallback dispatches *to* BinaryBackend, comparing its
  output against a direct BinaryBackend call is tautological. These
  aren't correctness tests — they're a coverage harness to make sure
  every via_binary branch compiles, transfers, and rewraps without
  error, and that the dtype/shape of the result matches the Nx
  contract for the op.

  The cost of these fallbacks (a full round-trip to CPU per call) is
  why the ops they guard are on the roadmap for native translation;
  keeping the smoke coverage here lets us delete each test as the
  corresponding native path lands.
  """

  use ExUnit.Case, async: true

  defp emily(list, type \\ {:f, 32}) do
    Nx.tensor(list, type: type, backend: Emily.Backend)
  end

  defp flat(t), do: Nx.to_flat_list(t)

  describe "indexing fallbacks" do
    test "put_slice routes through BinaryBackend" do
      t = emily([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      u = emily([[10.0, 20.0]])

      result = Nx.put_slice(t, [0, 1], u)

      assert Nx.shape(result) == {2, 3}
      assert flat(result) == [1.0, 10.0, 20.0, 4.0, 5.0, 6.0]
    end

    test "gather with multi-axis indices routes through BinaryBackend" do
      t = emily([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      # Multi-axis index: each index selects across axes [0, 1].
      idx = Nx.tensor([[0, 0], [1, 1], [2, 0]], backend: Emily.Backend)

      result = Nx.gather(t, idx, axes: [0, 1])

      assert flat(result) == [1.0, 4.0, 5.0]
    end

    test "indexed_add routes through BinaryBackend" do
      t = emily([1.0, 2.0, 3.0, 4.0])
      idx = Nx.tensor([[0], [2]], backend: Emily.Backend)
      upd = emily([10.0, 100.0])

      result = Nx.indexed_add(t, idx, upd)

      assert flat(result) == [11.0, 2.0, 103.0, 4.0]
    end

    test "indexed_put routes through BinaryBackend" do
      t = emily([1.0, 2.0, 3.0, 4.0])
      idx = Nx.tensor([[0], [2]], backend: Emily.Backend)
      upd = emily([99.0, 77.0])

      result = Nx.indexed_put(t, idx, upd)

      assert flat(result) == [99.0, 2.0, 77.0, 4.0]
    end
  end

  describe "reduce fallbacks" do
    test "reduce with a custom accumulator function" do
      t = emily([1.0, 2.0, 3.0, 4.0])
      acc = Nx.tensor(0.0, backend: Emily.Backend)

      result = Nx.reduce(t, acc, fn x, a -> Nx.add(x, a) end)

      assert Nx.to_number(result) == 10.0
    end

    test "window_reduce with a custom accumulator function" do
      t = emily([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      acc = Nx.tensor(0.0, backend: Emily.Backend)

      result =
        Nx.window_reduce(t, acc, {1, 2}, [strides: [1, 1]], fn x, a -> Nx.max(x, a) end)

      assert Nx.shape(result) == {2, 2}
      assert flat(result) == [2.0, 3.0, 5.0, 6.0]
    end
  end

  describe "window reductions" do
    setup do
      %{tensor: emily([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])}
    end

    test "window_sum", %{tensor: t} do
      result = Nx.window_sum(t, {1, 2})
      assert flat(result) == [3.0, 5.0, 7.0, 11.0, 13.0, 15.0]
    end

    test "window_product", %{tensor: t} do
      result = Nx.window_product(t, {1, 2})
      assert flat(result) == [2.0, 6.0, 12.0, 30.0, 42.0, 56.0]
    end

    test "window_max", %{tensor: t} do
      result = Nx.window_max(t, {1, 2})
      assert flat(result) == [2.0, 3.0, 4.0, 6.0, 7.0, 8.0]
    end

    test "window_min", %{tensor: t} do
      result = Nx.window_min(t, {1, 2})
      assert flat(result) == [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    end
  end

  describe "window scatter" do
    # Scatter selects the argmax/argmin within each window and scatters
    # the corresponding `source` entry into the output. Small inputs;
    # we're only checking the via_binary path runs clean.
    test "window_scatter_max" do
      t = emily([[1.0, 2.0], [3.0, 4.0]])
      source = emily([[5.0]])
      init = Nx.tensor(0.0, backend: Emily.Backend)

      result = Nx.window_scatter_max(t, source, init, {2, 2}, strides: [1, 1])

      assert Nx.shape(result) == {2, 2}
      # Max element is 4.0 (bottom-right); that position receives 5.0.
      assert flat(result) == [0.0, 0.0, 0.0, 5.0]
    end

    test "window_scatter_min" do
      t = emily([[1.0, 2.0], [3.0, 4.0]])
      source = emily([[5.0]])
      init = Nx.tensor(0.0, backend: Emily.Backend)

      result = Nx.window_scatter_min(t, source, init, {2, 2}, strides: [1, 1])

      # Min element is 1.0 (top-left).
      assert flat(result) == [5.0, 0.0, 0.0, 0.0]
    end
  end

  describe "native linalg" do
    test "lu returns (p, l, u) and reconstructs input" do
      t = emily([[2.0, 1.0], [1.0, 3.0]])
      {p, l, u} = Nx.LinAlg.lu(t)

      assert Nx.shape(p) == {2, 2}
      assert Nx.shape(l) == {2, 2}
      assert Nx.shape(u) == {2, 2}

      # Round-trip check: P * L * U ≈ original (within f32 tolerance).
      reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
      assert_in_delta Nx.to_number(reconstructed[0][0]), 2.0, 1.0e-4
      assert_in_delta Nx.to_number(reconstructed[1][1]), 3.0, 1.0e-4
    end

    test "svd returns (u, s, vt) with correct singular values" do
      t = emily([[3.0, 0.0], [0.0, 4.0]])
      {u, s, vt} = Nx.LinAlg.svd(t)

      assert Nx.shape(u) == {2, 2}
      assert Nx.shape(s) == {2}
      assert Nx.shape(vt) == {2, 2}

      # Singular values of a positive diagonal are its entries, sorted.
      [s0, s1] = flat(s)
      assert_in_delta max(s0, s1), 4.0, 1.0e-4
      assert_in_delta min(s0, s1), 3.0, 1.0e-4
    end

    test "svd with full_matrices?: false returns reduced dimensions" do
      # 2×3 matrix: reduced SVD gives U={2,2}, S={2}, Vt={2,3}
      t = emily([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
      {u, s, vt} = Nx.LinAlg.svd(t, full_matrices?: false)

      assert Nx.shape(u) == {2, 2}
      assert Nx.shape(s) == {2}
      assert Nx.shape(vt) == {2, 3}
    end

    test "triangular_solve: lower-triangular Lx = b" do
      l = emily([[2.0, 0.0], [1.0, 3.0]])
      b = emily([2.0, 4.0])

      x = Nx.LinAlg.triangular_solve(l, b)

      assert_in_delta Nx.to_number(x[0]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(x[1]), 1.0, 1.0e-4
    end

    test "triangular_solve: upper-triangular with lower: false" do
      u = emily([[2.0, 1.0], [0.0, 3.0]])
      b = emily([5.0, 3.0])

      x = Nx.LinAlg.triangular_solve(u, b, lower: false)

      assert_in_delta Nx.to_number(x[0]), 2.0, 1.0e-4
      assert_in_delta Nx.to_number(x[1]), 1.0, 1.0e-4
    end

    test "triangular_solve: left_side: false" do
      # Solve X A = B where A is lower triangular.
      a = emily([[1.0, 0.0], [2.0, 3.0]])
      b = emily([[1.0, 2.0], [3.0, 3.0]])

      x = Nx.LinAlg.triangular_solve(a, b, left_side: false)
      # Verify X A ≈ B
      reconstructed = Nx.dot(x, a)
      assert_in_delta Nx.to_number(reconstructed[0][0]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(reconstructed[1][1]), 3.0, 1.0e-4
    end

    test "triangular_solve: transform_a: :transpose" do
      # Solve A^T x = b with A lower-triangular
      a = emily([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
      b = emily([1.0, 2.0, 1.0])

      x = Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose, lower: false)

      ref =
        Nx.LinAlg.triangular_solve(
          a |> Nx.backend_transfer(Nx.BinaryBackend),
          b |> Nx.backend_transfer(Nx.BinaryBackend),
          transform_a: :transpose,
          lower: false
        )

      for i <- 0..2 do
        assert_in_delta Nx.to_number(x[i]), Nx.to_number(ref[i]), 1.0e-4
      end
    end

    test "triangular_solve: left_side: false + transform_a: :transpose" do
      # Solve X A^T = B
      a = emily([[1.0, 0.0], [2.0, 3.0]])
      b = emily([[1.0, 2.0], [3.0, 3.0]])

      x = Nx.LinAlg.triangular_solve(a, b, left_side: false, transform_a: :transpose)

      ref =
        Nx.LinAlg.triangular_solve(
          a |> Nx.backend_transfer(Nx.BinaryBackend),
          b |> Nx.backend_transfer(Nx.BinaryBackend),
          left_side: false,
          transform_a: :transpose
        )

      for i <- 0..1, j <- 0..1 do
        assert_in_delta Nx.to_number(x[i][j]), Nx.to_number(ref[i][j]), 1.0e-4
      end
    end

    test "qr (reduced) returns Q * R ≈ A" do
      t = emily([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)

      assert Nx.shape(q) == {3, 2}
      assert Nx.shape(r) == {2, 2}

      # Q * R ≈ A
      reconstructed = Nx.dot(q, r)
      assert_in_delta Nx.to_number(reconstructed[0][0]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(reconstructed[2][1]), 6.0, 1.0e-4
    end

    test "qr (complete) falls back correctly" do
      t = emily([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {q, r} = Nx.LinAlg.qr(t, mode: :complete)

      assert Nx.shape(q) == {3, 3}
      assert Nx.shape(r) == {3, 2}
    end

    test "cholesky: L * L^T ≈ A for an SPD matrix" do
      # A = [[4, 2], [2, 3]] — symmetric positive definite
      t = emily([[4.0, 2.0], [2.0, 3.0]])
      l = Nx.LinAlg.cholesky(t)

      assert Nx.shape(l) == {2, 2}

      reconstructed = Nx.dot(l, Nx.transpose(l))
      assert_in_delta Nx.to_number(reconstructed[0][0]), 4.0, 1.0e-4
      assert_in_delta Nx.to_number(reconstructed[1][1]), 3.0, 1.0e-4
    end

    test "eigh: eigenvalues match known values" do
      # A = [[2, 1], [1, 3]] — eigenvalues: (5 ± √5)/2 ≈ 1.382, 3.618
      t = emily([[2.0, 1.0], [1.0, 3.0]])
      {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t)

      assert Nx.shape(eigenvals) == {2}
      assert Nx.shape(eigenvecs) == {2, 2}

      vals = eigenvals |> Nx.sort() |> Nx.to_flat_list()
      assert_in_delta Enum.at(vals, 0), (5.0 - :math.sqrt(5)) / 2, 1.0e-3
      assert_in_delta Enum.at(vals, 1), (5.0 + :math.sqrt(5)) / 2, 1.0e-3
    end

    test "solve: Ax = b" do
      a = emily([[1.0, 2.0], [3.0, 4.0]])
      b = emily([5.0, 11.0])

      x = Nx.LinAlg.solve(a, b)
      assert_in_delta Nx.to_number(x[0]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(x[1]), 2.0, 1.0e-4
    end

    test "determinant via native lu for 3×3 matrix" do
      # det([[1, 2, 3], [0, 1, 4], [5, 6, 0]]) = 1
      t = emily([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
      d = Nx.LinAlg.determinant(t)
      assert_in_delta Nx.to_number(d), 1.0, 1.0e-3
    end
  end

  describe "forced fallback branches" do
    # Batched dot with integer operands: MLX matmul is float-only, so
    # `Emily.Backend.dot/7` routes integer-batched calls through
    # BinaryBackend. Float-batched path is covered by the property
    # tests in backend_test.exs.
    test "batched dot with s32 operands falls back" do
      a = Nx.iota({2, 3, 4}, type: {:s, 32}, backend: Emily.Backend)
      b = Nx.iota({2, 4, 5}, type: {:s, 32}, backend: Emily.Backend)

      emily = Nx.dot(a, [2], [0], b, [1], [0])

      ref_a = Nx.iota({2, 3, 4}, type: {:s, 32}, backend: Nx.BinaryBackend)
      ref_b = Nx.iota({2, 4, 5}, type: {:s, 32}, backend: Nx.BinaryBackend)
      ref = Nx.dot(ref_a, [2], [0], ref_b, [1], [0])

      assert Nx.shape(emily) == {2, 3, 5}
      assert flat(emily) == Nx.to_flat_list(ref)
    end

    # Interior-axis cumulative: MLX's cumulative kernels raise on
    # some 4-D+ shape factorings, so the backend routes interior-axis
    # cumulation through BinaryBackend. Last-axis stays native.
    test "cumulative_sum on an interior axis falls back" do
      t = Nx.iota({2, 3, 4}, type: {:f, 32}, backend: Emily.Backend)
      result = Nx.cumulative_sum(t, axis: 1)

      ref = Nx.iota({2, 3, 4}, type: {:f, 32}, backend: Nx.BinaryBackend)
      ref_result = Nx.cumulative_sum(ref, axis: 1)

      assert flat(result) == Nx.to_flat_list(ref_result)
    end

    test "cumulative_product on an interior axis falls back" do
      t = emily([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      result = Nx.cumulative_product(t, axis: 1)
      assert Nx.shape(result) == {2, 2, 2}
    end

    test "cumulative_max on an interior axis falls back" do
      t = emily([[[1.0, 4.0], [3.0, 2.0]], [[5.0, 8.0], [7.0, 6.0]]])
      result = Nx.cumulative_max(t, axis: 1)
      assert Nx.shape(result) == {2, 2, 2}
    end

    test "cumulative_min on an interior axis falls back" do
      t = emily([[[1.0, 4.0], [3.0, 2.0]], [[5.0, 8.0], [7.0, 6.0]]])
      result = Nx.cumulative_min(t, axis: 1)
      assert Nx.shape(result) == {2, 2, 2}
    end

    # Conv with `batch_group_size > 1`: MLX `conv_general` has no batch-group
    # parameter, so Emily falls back to BinaryBackend for this rare path. The
    # common `batch_group_size: 1` case is covered natively in
    # `backend_conv_test.exs`.
    test "conv with batch_group_size > 1 falls back" do
      # Splitting batch=4 into 2 groups; each produces 1 output filter from
      # its half of the input channels.
      input = Nx.iota({4, 2, 3, 3}, type: {:f, 32}, backend: Emily.Backend)
      kernel = Nx.iota({2, 2, 2, 2}, type: {:f, 32}, backend: Emily.Backend)

      result = Nx.conv(input, kernel, batch_group_size: 2)

      ref_input = Nx.iota({4, 2, 3, 3}, type: {:f, 32}, backend: Nx.BinaryBackend)
      ref_kernel = Nx.iota({2, 2, 2, 2}, type: {:f, 32}, backend: Nx.BinaryBackend)
      ref = Nx.conv(ref_input, ref_kernel, batch_group_size: 2)

      assert Nx.shape(result) == Nx.shape(ref)
      assert flat(result) == Nx.to_flat_list(ref)
    end
  end
end
