defmodule Emily.BackendWindowTest do
  @moduledoc """
  Oracle tests for M17's native window reductions (`window_sum`,
  `window_max`, `window_min`, `window_product`). Each case runs the
  same op on `Emily.Backend` and `Nx.BinaryBackend` and asserts
  element-wise agreement.

  Covers rank 1-4, `:valid`/`:same`/explicit padding, strided
  windows, and window dilation > 1. Covers f32, bf16, and integer
  dtypes — the integer path exercises the dtype-specific identity
  branch for window_max/window_min (no `-inf` for `{:s, _}` / `{:u,
  _}`).
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3]

  defp emily(tensor), do: Nx.backend_transfer(tensor, Emily.Backend)
  defp bin(tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  # Deterministic float input; scaled so window_product stays in f32
  # range for small kernels.
  defp fixt(shape, type \\ {:f, 32}) do
    size = shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

    Nx.iota({size}, type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.multiply(0.3)
    |> Nx.add(1.0)
    |> Nx.reshape(shape)
    |> Nx.as_type(type)
  end

  defp run(nx_fun, tensor, window_shape, opts, tol \\ 1.0e-5) do
    emily_result = apply(Nx, nx_fun, [emily(tensor), window_shape, opts])
    ref_result = apply(Nx, nx_fun, [bin(tensor), window_shape, opts])
    assert_close(emily_result, ref_result, tol: tol)
  end

  describe "window_sum" do
    test "1-D kernel" do
      t = fixt({8})
      run(:window_sum, t, {3}, [])
    end

    test "2-D kernel, :valid padding" do
      t = fixt({4, 5})
      run(:window_sum, t, {2, 2}, [])
    end

    test "2-D kernel, :same padding" do
      t = fixt({4, 5})
      run(:window_sum, t, {2, 2}, padding: :same)
    end

    test "2-D kernel with stride 2" do
      t = fixt({6, 6})
      run(:window_sum, t, {2, 2}, strides: [2, 2])
    end

    test "2-D kernel with explicit asymmetric padding" do
      t = fixt({4, 4})
      run(:window_sum, t, {2, 2}, padding: [{1, 0}, {0, 2}])
    end

    test "3-D kernel over 3-D tensor" do
      t = fixt({3, 4, 5})
      run(:window_sum, t, {2, 2, 3}, [])
    end

    test "4-D NCHW-shaped input, kernel on spatial axes only" do
      t = fixt({2, 3, 5, 5})
      run(:window_sum, t, {1, 1, 3, 3}, strides: [1, 1, 2, 2])
    end

    test "window dilations > 1" do
      t = fixt({1, 1, 7, 7})
      run(:window_sum, t, {1, 1, 2, 2}, window_dilations: [1, 1, 2, 2])
    end

    test "bf16 input (wider tolerance)" do
      t = fixt({4, 4}, {:bf, 16})
      run(:window_sum, t, {2, 2}, [], 1.0e-2)
    end

    test "s32 input" do
      t =
        Nx.iota({4, 5}, type: {:s, 32}, backend: Nx.BinaryBackend)
        |> Nx.multiply(3)

      run(:window_sum, t, {2, 2}, [], 0.0)
    end
  end

  describe "window_max / window_min" do
    test "window_max 2-D" do
      t = fixt({4, 5})
      run(:window_max, t, {2, 2}, [])
      run(:window_min, t, {2, 2}, [])
    end

    test "window_max with :same padding uses -inf fill" do
      t = fixt({3, 3})
      run(:window_max, t, {2, 2}, padding: :same)
      run(:window_min, t, {2, 2}, padding: :same)
    end

    test "window_max on negative inputs (fill must be lower than any valid value)" do
      t = fixt({3, 4}) |> Nx.negate()
      run(:window_max, t, {1, 2}, padding: :same)
    end

    test "window_max on s32 (dtype-specific min fill)" do
      t =
        Nx.iota({4, 4}, type: {:s, 32}, backend: Nx.BinaryBackend)
        |> Nx.subtract(8)

      run(:window_max, t, {2, 2}, [padding: :same], 0.0)
      run(:window_min, t, {2, 2}, [padding: :same], 0.0)
    end

    test "window_max on u8 (dtype-specific max/min fill)" do
      t =
        Nx.iota({5, 5}, type: {:u, 8}, backend: Nx.BinaryBackend)
        |> Nx.remainder(200)

      run(:window_max, t, {2, 2}, padding: :same)
      run(:window_min, t, {2, 2}, padding: :same)
    end

    test "strides + dilation" do
      t = fixt({1, 1, 8, 8})
      run(:window_max, t, {1, 1, 2, 2}, strides: [1, 1, 2, 2], window_dilations: [1, 1, 2, 2])
    end
  end

  describe "window_product" do
    test "2-D kernel, small values to avoid overflow" do
      t =
        Nx.iota({3, 4}, type: {:f, 32}, backend: Nx.BinaryBackend)
        |> Nx.multiply(0.2)
        |> Nx.add(1.0)

      run(:window_product, t, {2, 2}, [])
    end

    test "identity (1.0) fill at padded boundary" do
      t =
        Nx.iota({2, 2}, type: {:f, 32}, backend: Nx.BinaryBackend)
        |> Nx.add(2.0)

      run(:window_product, t, {2, 2}, padding: :same)
    end
  end

  describe "no-op window (window_shape == {1, 1, ...})" do
    test "window_sum with trivial kernel preserves input" do
      t = fixt({3, 4})
      result = Nx.window_sum(emily(t), {1, 1}, [])
      assert_close(result, bin(t), tol: 1.0e-5)
    end
  end
end
