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

  describe "convolution fallback" do
    test "conv routes through BinaryBackend" do
      # {batch=1, channels=1, height=3, width=3} input, {out=1, in=1, 2, 2} kernel.
      input = Nx.iota({1, 1, 3, 3}, type: {:f, 32}, backend: Emily.Backend)
      kernel = emily([[[[1.0, 0.0], [0.0, 1.0]]]])

      result = Nx.conv(input, kernel)

      assert Nx.shape(result) == {1, 1, 2, 2}
      # Diagonal kernel: 0+4=4, 1+5=6, 3+7=10, 4+8=12.
      assert flat(result) == [4.0, 6.0, 10.0, 12.0]
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

  describe "linear algebra fallbacks" do
    test "lu returns (p, l, u) tuple via BinaryBackend" do
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

    test "svd returns (u, s, vt) tuple via BinaryBackend" do
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

    test "triangular_solve routes through BinaryBackend" do
      # L x = b with L lower-triangular; x should be [1, 1].
      l = emily([[2.0, 0.0], [1.0, 3.0]])
      b = emily([2.0, 4.0])

      x = Nx.LinAlg.triangular_solve(l, b)

      assert_in_delta Nx.to_number(x[0]), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(x[1]), 1.0, 1.0e-4
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
  end
end
