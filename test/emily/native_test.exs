defmodule Emily.NativeTest do
  @moduledoc """
  Unit tests for the Native NIF surface. Each NIF is called directly
  (no Backend, no Defn) with hand-computed expected outputs. See
  `test/emily_test.exs` for higher-level round-trip tests.
  """

  use ExUnit.Case, async: true

  alias Emily.Native

  # ---------- Helpers ----------

  defp f32(list, shape) when is_list(list) do
    bin = for x <- list, into: <<>>, do: <<x * 1.0::float-32-native>>
    Native.from_binary(bin, shape, {:f, 32})
  end

  defp f32_scalar(x), do: f32([x], [])

  defp to_f32_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<f::float-32-native <- bin>>, do: f
  end

  defp s32(list, shape) when is_list(list) do
    bin = for x <- list, into: <<>>, do: <<x::signed-integer-32-native>>
    Native.from_binary(bin, shape, {:s, 32})
  end

  defp to_s32_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<i::signed-integer-32-native <- bin>>, do: i
  end

  defp pred(list, shape) when is_list(list) do
    bin = for b <- list, into: <<>>, do: <<if(b, do: 1, else: 0)>>
    Native.from_binary(bin, shape, {:pred, 1})
  end

  defp to_pred_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<b::unsigned-integer-8 <- bin>>, do: b == 1
  end

  defp assert_close(actual, expected, tol \\ 1.0e-5)

  defp assert_close(actual, expected, tol) when is_list(actual) and is_list(expected) do
    assert length(actual) == length(expected),
           "length mismatch: #{inspect(actual)} vs #{inspect(expected)}"

    Enum.zip(actual, expected)
    |> Enum.each(fn {a, e} -> assert_close(a, e, tol) end)
  end

  defp assert_close(actual, expected, tol) when is_number(actual) and is_number(expected) do
    if abs(actual - expected) <= tol + tol * abs(expected) do
      :ok
    else
      flunk("expected #{expected}, got #{actual} (tol=#{tol})")
    end
  end

  # ---------- Creation ----------

  describe "creation" do
    test "zeros/2" do
      t = Native.zeros([2, 3], {:f, 32})
      assert Native.shape(t) == [2, 3]
      assert Native.dtype(t) == {:f, 32}
      assert to_f32_list(t) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "ones/2" do
      t = Native.ones([4], {:s, 32})
      assert to_s32_list(t) == [1, 1, 1, 1]
    end

    test "full/3 broadcasts a scalar value" do
      v = f32_scalar(3.5)
      t = Native.full([2, 2], v, {:f, 32})
      assert to_f32_list(t) == [3.5, 3.5, 3.5, 3.5]
    end

    test "arange/4" do
      t = Native.arange(0.0, 5.0, 1.0, {:s, 32})
      assert to_s32_list(t) == [0, 1, 2, 3, 4]
    end

    test "eye/4" do
      t = Native.eye(3, 3, 0, {:f, 32})
      assert to_f32_list(t) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  # ---------- Cast ----------

  describe "cast" do
    test "astype: f32 -> s32" do
      t = f32([1.2, -2.7, 3.5], [3])
      out = Native.astype(t, {:s, 32})
      assert Native.dtype(out) == {:s, 32}
      # MLX truncates toward zero on float->int cast.
      assert to_s32_list(out) == [1, -2, 3]
    end

    test "astype: s32 -> f32" do
      t = s32([1, 2, 3], [3])
      out = Native.astype(t, {:f, 32})
      assert Native.dtype(out) == {:f, 32}
      assert to_f32_list(out) == [1.0, 2.0, 3.0]
    end
  end

  # ---------- Unary ----------

  describe "unary elementwise" do
    test "negative" do
      assert to_f32_list(Native.negative(f32([1.0, -2.0, 3.0], [3]))) == [-1.0, 2.0, -3.0]
    end

    test "abs" do
      assert to_f32_list(Native.abs(f32([-1.5, 2.0, -0.0], [3]))) == [1.5, 2.0, 0.0]
    end

    test "sign" do
      assert to_f32_list(Native.sign(f32([-2.0, 0.0, 3.0], [3]))) == [-1.0, 0.0, 1.0]
    end

    test "floor / ceil / round" do
      x = f32([1.7, -1.7, 2.5], [3])
      assert to_f32_list(Native.floor(x)) == [1.0, -2.0, 2.0]
      assert to_f32_list(Native.ceil(x)) == [2.0, -1.0, 3.0]
      # MLX's round rounds-half-to-even on exact halves when decimals=0.
      assert to_f32_list(Native.round(x, 0)) == [2.0, -2.0, 2.0]
    end

    test "sqrt / rsqrt / square / reciprocal" do
      x = f32([4.0, 9.0], [2])
      assert to_f32_list(Native.sqrt(x)) == [2.0, 3.0]
      assert_close(to_f32_list(Native.rsqrt(x)), [0.5, 1.0 / 3.0])
      assert to_f32_list(Native.square(x)) == [16.0, 81.0]
      assert_close(to_f32_list(Native.reciprocal(x)), [0.25, 1.0 / 9.0])
    end

    test "exp / expm1 / log / log1p / log2 / log10" do
      x = f32([1.0, 2.0], [2])
      assert_close(to_f32_list(Native.exp(x)), [:math.exp(1.0), :math.exp(2.0)])
      assert_close(to_f32_list(Native.expm1(x)), [:math.exp(1.0) - 1.0, :math.exp(2.0) - 1.0])
      assert_close(to_f32_list(Native.log(x)), [0.0, :math.log(2.0)])
      assert_close(to_f32_list(Native.log1p(x)), [:math.log(2.0), :math.log(3.0)])
      assert_close(to_f32_list(Native.log2(x)), [0.0, 1.0])
      assert_close(to_f32_list(Native.log10(x)), [0.0, :math.log10(2.0)])
    end

    test "trig: sin / cos / tan" do
      x = f32([0.0, :math.pi() / 2], [2])
      assert_close(to_f32_list(Native.sin(x)), [0.0, 1.0], 1.0e-4)
      assert_close(to_f32_list(Native.cos(x)), [1.0, 0.0], 1.0e-4)
      assert_close(to_f32_list(Native.tan(f32([0.0], [1]))), [0.0])
    end

    test "inverse trig: arcsin / arccos / arctan" do
      assert_close(
        to_f32_list(Native.arcsin(f32([0.0, 1.0], [2]))),
        [0.0, :math.pi() / 2],
        1.0e-4
      )

      assert_close(
        to_f32_list(Native.arccos(f32([1.0, 0.0], [2]))),
        [0.0, :math.pi() / 2],
        1.0e-4
      )

      assert_close(
        to_f32_list(Native.arctan(f32([0.0, 1.0], [2]))),
        [0.0, :math.pi() / 4],
        1.0e-4
      )
    end

    test "hyperbolic: sinh / cosh / tanh and their inverses" do
      x = f32([0.0, 1.0], [2])
      assert_close(to_f32_list(Native.sinh(x)), [0.0, :math.sinh(1.0)])
      assert_close(to_f32_list(Native.cosh(x)), [1.0, :math.cosh(1.0)])
      assert_close(to_f32_list(Native.tanh(x)), [0.0, :math.tanh(1.0)])
      assert_close(to_f32_list(Native.arcsinh(f32([0.0], [1]))), [0.0])
      assert_close(to_f32_list(Native.arccosh(f32([1.0], [1]))), [0.0])
      assert_close(to_f32_list(Native.arctanh(f32([0.0], [1]))), [0.0])
    end

    test "sigmoid" do
      x = f32([0.0, 10.0, -10.0], [3])
      assert_close(to_f32_list(Native.sigmoid(x)), [0.5, 1.0, 0.0], 1.0e-4)
    end

    test "erf / erfinv" do
      assert_close(to_f32_list(Native.erf(f32([0.0], [1]))), [0.0], 1.0e-6)
      assert_close(to_f32_list(Native.erfinv(f32([0.0], [1]))), [0.0], 1.0e-6)
    end

    test "logical_not" do
      p = pred([true, false, true], [3])
      assert to_pred_list(Native.logical_not(p)) == [false, true, false]
    end

    test "bitwise_invert" do
      t = s32([0, -1, 5], [3])
      assert to_s32_list(Native.bitwise_invert(t)) == [-1, 0, -6]
    end

    test "isnan / isinf / isfinite" do
      x = f32([0.0, 1.0], [2])
      assert to_pred_list(Native.isnan(x)) == [false, false]
      assert to_pred_list(Native.isinf(x)) == [false, false]
      assert to_pred_list(Native.isfinite(x)) == [true, true]
    end

    test "stop_gradient is identity in forward pass" do
      x = f32([1.0, 2.0, 3.0], [3])
      assert to_f32_list(Native.stop_gradient(x)) == [1.0, 2.0, 3.0]
    end
  end

  # ---------- Binary ----------

  describe "binary arithmetic" do
    test "add / subtract / multiply / divide" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([10.0, 20.0, 30.0], [3])
      assert to_f32_list(Native.add(a, b)) == [11.0, 22.0, 33.0]
      assert to_f32_list(Native.subtract(a, b)) == [-9.0, -18.0, -27.0]
      assert to_f32_list(Native.multiply(a, b)) == [10.0, 40.0, 90.0]
      assert to_f32_list(Native.divide(b, a)) == [10.0, 10.0, 10.0]
    end

    test "floor_divide / remainder" do
      a = s32([7, 8, 9], [3])
      b = s32([2, 3, 4], [3])
      assert to_s32_list(Native.floor_divide(a, b)) == [3, 2, 2]
      assert to_s32_list(Native.remainder(a, b)) == [1, 2, 1]
    end

    test "power" do
      a = f32([2.0, 3.0], [2])
      b = f32([3.0, 2.0], [2])
      assert to_f32_list(Native.power(a, b)) == [8.0, 9.0]
    end

    test "maximum / minimum" do
      a = f32([1.0, 5.0, 3.0], [3])
      b = f32([4.0, 2.0, 3.0], [3])
      assert to_f32_list(Native.maximum(a, b)) == [4.0, 5.0, 3.0]
      assert to_f32_list(Native.minimum(a, b)) == [1.0, 2.0, 3.0]
    end

    test "logaddexp" do
      a = f32([0.0, 0.0], [2])
      b = f32([0.0, 1.0], [2])

      assert_close(to_f32_list(Native.logaddexp(a, b)), [
        :math.log(2.0),
        :math.log(1.0 + :math.exp(1.0))
      ])
    end

    test "arctan2" do
      assert_close(to_f32_list(Native.arctan2(f32([1.0], [1]), f32([1.0], [1]))), [:math.pi() / 4])
    end

    test "broadcasting: [3] + [1]" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([10.0], [1])
      assert to_f32_list(Native.add(a, b)) == [11.0, 12.0, 13.0]
    end
  end

  describe "comparisons" do
    test "equal / not_equal" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([1.0, 5.0, 3.0], [3])
      assert to_pred_list(Native.equal(a, b)) == [true, false, true]
      assert to_pred_list(Native.not_equal(a, b)) == [false, true, false]
    end

    test "less / less_equal / greater / greater_equal" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([2.0, 2.0, 2.0], [3])
      assert to_pred_list(Native.less(a, b)) == [true, false, false]
      assert to_pred_list(Native.less_equal(a, b)) == [true, true, false]
      assert to_pred_list(Native.greater(a, b)) == [false, false, true]
      assert to_pred_list(Native.greater_equal(a, b)) == [false, true, true]
    end
  end

  describe "logical" do
    test "logical_and / logical_or" do
      a = pred([true, true, false, false], [4])
      b = pred([true, false, true, false], [4])
      assert to_pred_list(Native.logical_and(a, b)) == [true, false, false, false]
      assert to_pred_list(Native.logical_or(a, b)) == [true, true, true, false]
    end
  end

  describe "bitwise" do
    test "and / or / xor" do
      a = s32([0b1100, 0b1010], [2])
      b = s32([0b1010, 0b0110], [2])
      assert to_s32_list(Native.bitwise_and(a, b)) == [0b1000, 0b0010]
      assert to_s32_list(Native.bitwise_or(a, b)) == [0b1110, 0b1110]
      assert to_s32_list(Native.bitwise_xor(a, b)) == [0b0110, 0b1100]
    end

    test "left_shift / right_shift" do
      a = s32([1, 16], [2])
      b = s32([3, 2], [2])
      assert to_s32_list(Native.left_shift(a, b)) == [8, 64]
      assert to_s32_list(Native.right_shift(a, b)) == [0, 4]
    end
  end

  # ---------- Reductions ----------

  describe "reductions" do
    test "sum/mean/prod over all axes" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      assert to_f32_list(Native.sum(x, [0, 1], false)) == [10.0]
      assert to_f32_list(Native.mean(x, [0, 1], false)) == [2.5]
      assert to_f32_list(Native.prod(x, [0, 1], false)) == [24.0]
    end

    test "sum with axes + keepdims" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      # sum over axis 1
      r = Native.sum(x, [1], false)
      assert Native.shape(r) == [2]
      assert to_f32_list(r) == [3.0, 7.0]

      r_keep = Native.sum(x, [1], true)
      assert Native.shape(r_keep) == [2, 1]
    end

    test "max / min" do
      x = f32([1.0, 5.0, 3.0, 2.0], [4])
      assert to_f32_list(Native.max(x, [0], false)) == [5.0]
      assert to_f32_list(Native.min(x, [0], false)) == [1.0]
    end

    test "all / any" do
      p = pred([true, true, false], [3])
      assert to_pred_list(Native.all(p, [0], false)) == [false]
      assert to_pred_list(Native.any(p, [0], false)) == [true]
    end

    test "logsumexp" do
      x = f32([1.0, 2.0, 3.0], [3])
      expected = :math.log(:math.exp(1.0) + :math.exp(2.0) + :math.exp(3.0))
      assert_close(to_f32_list(Native.logsumexp(x, [0], false)), [expected])
    end

    test "argmax / argmin" do
      x = f32([1.0, 5.0, 3.0], [3])
      assert to_s32_list(Native.argmax(x, 0, false)) == [1]
      assert to_s32_list(Native.argmin(x, 0, false)) == [0]
    end

    test "var / std" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      # var with ddof=0 => population variance = 1.25
      assert_close(to_f32_list(Native.var(x, [0], false, 0)), [1.25])
      assert_close(to_f32_list(Native.std(x, [0], false, 0)), [:math.sqrt(1.25)])
    end

    test "cumulative: cumsum / cumprod" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      # inclusive, not reversed
      assert to_f32_list(Native.cumsum(x, 0, false, true)) == [1.0, 3.0, 6.0, 10.0]
      assert to_f32_list(Native.cumprod(x, 0, false, true)) == [1.0, 2.0, 6.0, 24.0]
    end

    test "cumulative: cummax / cummin" do
      x = f32([3.0, 1.0, 4.0, 1.0, 5.0], [5])
      assert to_f32_list(Native.cummax(x, 0, false, true)) == [3.0, 3.0, 4.0, 4.0, 5.0]
      assert to_f32_list(Native.cummin(x, 0, false, true)) == [3.0, 1.0, 1.0, 1.0, 1.0]
    end
  end

  # ---------- Shape ----------

  describe "shape manipulation" do
    test "reshape" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
      r = Native.reshape(x, [2, 3])
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "transpose" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      r = Native.transpose(x, [1, 0])
      assert Native.shape(r) == [3, 2]
      assert to_f32_list(r) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    end

    test "squeeze / expand_dims" do
      x = f32([1.0, 2.0, 3.0], [1, 3, 1])
      s = Native.squeeze(x, [0, 2])
      assert Native.shape(s) == [3]
      e = Native.expand_dims(s, [0])
      assert Native.shape(e) == [1, 3]
    end

    test "broadcast_to" do
      x = f32([1.0, 2.0, 3.0], [3])
      r = Native.broadcast_to(x, [2, 3])
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    end

    test "concatenate / stack" do
      a = f32([1.0, 2.0], [2])
      b = f32([3.0, 4.0], [2])
      c = Native.concatenate([a, b], 0)
      assert Native.shape(c) == [4]
      assert to_f32_list(c) == [1.0, 2.0, 3.0, 4.0]

      s = Native.stack([a, b], 0)
      assert Native.shape(s) == [2, 2]
      assert to_f32_list(s) == [1.0, 2.0, 3.0, 4.0]
    end

    test "flatten" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      r = Native.flatten(x, 0, -1)
      assert Native.shape(r) == [4]
    end

    test "tile" do
      x = f32([1.0, 2.0], [2])
      r = Native.tile(x, [3])
      assert to_f32_list(r) == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    end

    test "swapaxes" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      r = Native.swapaxes(x, 0, 1)
      assert to_f32_list(r) == [1.0, 3.0, 2.0, 4.0]
    end

    test "pad" do
      x = f32([1.0, 2.0, 3.0], [3])
      zero = f32_scalar(0.0)
      r = Native.pad(x, [0], [1], [2], zero)
      assert Native.shape(r) == [6]
      assert to_f32_list(r) == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    end

    test "repeat" do
      x = f32([1.0, 2.0], [2])
      r = Native.repeat(x, 2, 0)
      assert to_f32_list(r) == [1.0, 1.0, 2.0, 2.0]
    end
  end

  # ---------- Indexing ----------

  describe "indexing" do
    test "slice" do
      x = f32(Enum.to_list(1..12) |> Enum.map(&(&1 * 1.0)), [3, 4])
      r = Native.slice(x, [0, 1], [2, 3], [1, 1])
      assert Native.shape(r) == [2, 2]
      assert to_f32_list(r) == [2.0, 3.0, 6.0, 7.0]
    end

    test "take" do
      x = f32([10.0, 20.0, 30.0, 40.0], [4])
      idx = s32([0, 2, 3], [3])
      r = Native.take(x, idx, 0)
      assert to_f32_list(r) == [10.0, 30.0, 40.0]
    end

    test "where" do
      cond_t = pred([true, false, true], [3])
      x = f32([1.0, 2.0, 3.0], [3])
      y = f32([10.0, 20.0, 30.0], [3])
      r = Native.where(cond_t, x, y)
      assert to_f32_list(r) == [1.0, 20.0, 3.0]
    end
  end

  # ---------- Linalg ----------

  describe "linalg" do
    test "matmul: 2x3 @ 3x2" do
      a = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      b = f32([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
      r = Native.matmul(a, b)
      assert Native.shape(r) == [2, 2]
      # [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
      assert to_f32_list(r) == [58.0, 64.0, 139.0, 154.0]
    end

    test "tensordot with axes" do
      a = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      b = f32([5.0, 6.0, 7.0, 8.0], [2, 2])
      # contract last axis of a with first of b (= matmul)
      r = Native.tensordot(a, b, [1], [0])
      assert to_f32_list(r) == [19.0, 22.0, 43.0, 50.0]
    end

    test "outer" do
      a = f32([1.0, 2.0], [2])
      b = f32([10.0, 20.0, 30.0], [3])
      r = Native.outer(a, b)
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    end

    test "inner of 1-D vectors = dot product" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([4.0, 5.0, 6.0], [3])
      r = Native.inner(a, b)
      assert to_f32_list(r) == [32.0]
    end
  end

  # ---------- Lifecycle ----------

  describe "lifecycle under load" do
    test "chained lazy ops survive GC before eval" do
      a = f32([1.0, 2.0, 3.0, 4.0], [4])
      b = f32([10.0, 20.0, 30.0, 40.0], [4])
      c = Native.add(a, b)
      d = Native.multiply(c, c)

      :erlang.garbage_collect()
      assert to_f32_list(d) == [121.0, 484.0, 1089.0, 1936.0]
    end
  end
end
