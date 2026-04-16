defmodule Emily.DtypeMatrixTest do
  @moduledoc """
  Smoke matrix: for each class of op, exercise every supported dtype
  at least once. This is a shallow liveness check — we don't verify
  numerical correctness, only that each dtype × op combination runs to
  completion and produces an output tensor with a sensible shape.
  """

  use ExUnit.Case, async: true

  alias Emily.Native

  @float_dtypes [{:f, 32}, {:f, 16}, {:bf, 16}]
  @int_dtypes [
    {:s, 8},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:u, 8},
    {:u, 16},
    {:u, 32}
  ]
  @all_numeric @float_dtypes ++ @int_dtypes

  defp any_dtype(shape, dtype) do
    # Use `ones` as the universal constructor — supported for every
    # numeric dtype MLX cares about.
    Native.ones(shape, dtype, -1)
  end

  describe "creation" do
    test "zeros / ones across dtypes" do
      for dtype <- @all_numeric do
        z = Native.zeros([4], dtype, -1)
        o = Native.ones([4], dtype, -1)
        assert Native.shape(z) == [4]
        assert Native.dtype(z) == dtype
        assert Native.shape(o) == [4]
        assert Native.dtype(o) == dtype
      end
    end
  end

  describe "cast" do
    test "astype across the numeric matrix" do
      for src <- @all_numeric, dst <- @all_numeric do
        t = any_dtype([4], src)
        out = Native.astype(t, dst, -1)

        assert Native.dtype(out) == dst,
               "astype #{inspect(src)} -> #{inspect(dst)} yielded #{inspect(Native.dtype(out))}"

        assert Native.shape(out) == [4]
      end
    end
  end

  describe "unary (float-only)" do
    # Transcendental ops only make sense for floats.
    @float_unary [:exp, :log, :sin, :cos, :tanh, :sigmoid, :sqrt, :rsqrt]

    test "transcendentals run across float dtypes" do
      for dtype <- @float_dtypes, op <- @float_unary do
        t = any_dtype([4], dtype)
        out = apply(Native, op, [t, -1])

        assert Native.dtype(out) == dtype,
               "#{op} on #{inspect(dtype)} yielded #{inspect(Native.dtype(out))}"

        assert Native.shape(out) == [4]
      end
    end
  end

  describe "unary (numeric)" do
    @num_unary [:negative, :abs, :square]

    test "simple arithmetic unaries run across numeric dtypes" do
      for dtype <- @all_numeric, op <- @num_unary do
        # Unsigned + negative is nonsensical; skip that pair.
        if op == :negative and match?({:u, _}, dtype) do
          :skip
        else
          t = any_dtype([4], dtype)
          out = apply(Native, op, [t, -1])
          assert Native.dtype(out) == dtype
          assert Native.shape(out) == [4]
        end
      end
    end
  end

  describe "binary arithmetic" do
    @bin_arith [:add, :subtract, :multiply, :maximum, :minimum]

    test "elementwise binaries run across numeric dtypes" do
      for dtype <- @all_numeric, op <- @bin_arith do
        a = any_dtype([4], dtype)
        b = any_dtype([4], dtype)
        out = apply(Native, op, [a, b, -1])
        assert Native.dtype(out) == dtype
        assert Native.shape(out) == [4]
      end
    end
  end

  describe "reductions" do
    test "sum / mean / max / min reduce to scalar across float dtypes" do
      for dtype <- @float_dtypes, op <- [:sum, :mean, :max, :min] do
        t = any_dtype([2, 3], dtype)
        out = apply(Native, op, [t, [0, 1], false, -1])
        assert Native.dtype(out) == dtype
        assert Native.shape(out) == []
      end
    end
  end

  describe "comparisons yield pred" do
    test "equal / less / greater always produce {:pred, 1}" do
      for dtype <- @all_numeric, op <- [:equal, :less, :greater] do
        a = any_dtype([4], dtype)
        b = any_dtype([4], dtype)
        out = apply(Native, op, [a, b, -1])
        assert Native.dtype(out) == {:pred, 1}
        assert Native.shape(out) == [4]
      end
    end
  end

  describe "unsupported dtypes raise clearly" do
    test "f64 is rejected with a pointer at f32" do
      assert_raise ArgumentError, ~r/unsupported dtype/, fn ->
        Native.zeros([2], {:f, 64}, -1)
      end
    end
  end
end
