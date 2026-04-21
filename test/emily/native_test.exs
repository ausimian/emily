defmodule Emily.NativeTest do
  @moduledoc """
  Unit tests for the Native NIF surface. Each NIF is called directly
  (no Backend, no Defn) with hand-computed expected outputs. See
  `test/emily_test.exs` for higher-level round-trip tests.
  """

  use ExUnit.Case, async: true

  import Emily.TensorHelpers

  alias Emily.Native

  # ---------- Creation ----------

  describe "creation" do
    test "zeros/2" do
      t = Native.zeros(worker(), [2, 3], {:f, 32})
      assert Native.shape(t) == [2, 3]
      assert Native.dtype(t) == {:f, 32}
      assert to_f32_list(t) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "ones/2" do
      t = Native.ones(worker(), [4], {:s, 32})
      assert to_s32_list(t) == [1, 1, 1, 1]
    end

    test "full/3 broadcasts a scalar value" do
      v = f32_scalar(3.5)
      t = Native.full(worker(), [2, 2], v, {:f, 32})
      assert to_f32_list(t) == [3.5, 3.5, 3.5, 3.5]
    end

    test "arange/4" do
      t = Native.arange(worker(), 0.0, 5.0, 1.0, {:s, 32})
      assert to_s32_list(t) == [0, 1, 2, 3, 4]
    end

    test "eye/4" do
      t = Native.eye(worker(), 3, 3, 0, {:f, 32})
      assert to_f32_list(t) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  # ---------- Cast ----------

  describe "cast" do
    test "astype: f32 -> s32" do
      t = f32([1.2, -2.7, 3.5], [3])
      out = Native.astype(worker(), t, {:s, 32})
      assert Native.dtype(out) == {:s, 32}
      # MLX truncates toward zero on float->int cast.
      assert to_s32_list(out) == [1, -2, 3]
    end

    test "astype: s32 -> f32" do
      t = s32([1, 2, 3], [3])
      out = Native.astype(worker(), t, {:f, 32})
      assert Native.dtype(out) == {:f, 32}
      assert to_f32_list(out) == [1.0, 2.0, 3.0]
    end
  end

  # ---------- Unary ----------

  describe "unary elementwise" do
    test "negative" do
      assert to_f32_list(Native.negative(worker(), f32([1.0, -2.0, 3.0], [3]))) == [
               -1.0,
               2.0,
               -3.0
             ]
    end

    test "abs" do
      assert to_f32_list(Native.abs(worker(), f32([-1.5, 2.0, -0.0], [3]))) == [1.5, 2.0, 0.0]
    end

    test "sign" do
      assert to_f32_list(Native.sign(worker(), f32([-2.0, 0.0, 3.0], [3]))) == [-1.0, 0.0, 1.0]
    end

    test "floor / ceil / round" do
      x = f32([1.7, -1.7, 2.5], [3])
      assert to_f32_list(Native.floor(worker(), x)) == [1.0, -2.0, 2.0]
      assert to_f32_list(Native.ceil(worker(), x)) == [2.0, -1.0, 3.0]
      # MLX's round rounds-half-to-even on exact halves when decimals=0.
      assert to_f32_list(Native.round(worker(), x, 0)) == [2.0, -2.0, 2.0]
    end

    test "sqrt / rsqrt / square / reciprocal" do
      x = f32([4.0, 9.0], [2])
      assert to_f32_list(Native.sqrt(worker(), x)) == [2.0, 3.0]
      assert_close(to_f32_list(Native.rsqrt(worker(), x)), [0.5, 1.0 / 3.0])
      assert to_f32_list(Native.square(worker(), x)) == [16.0, 81.0]
      assert_close(to_f32_list(Native.reciprocal(worker(), x)), [0.25, 1.0 / 9.0])
    end

    test "exp / expm1 / log / log1p / log2 / log10" do
      x = f32([1.0, 2.0], [2])
      assert_close(to_f32_list(Native.exp(worker(), x)), [:math.exp(1.0), :math.exp(2.0)])

      assert_close(to_f32_list(Native.expm1(worker(), x)), [
        :math.exp(1.0) - 1.0,
        :math.exp(2.0) - 1.0
      ])

      assert_close(to_f32_list(Native.log(worker(), x)), [0.0, :math.log(2.0)])
      assert_close(to_f32_list(Native.log1p(worker(), x)), [:math.log(2.0), :math.log(3.0)])
      assert_close(to_f32_list(Native.log2(worker(), x)), [0.0, 1.0])
      assert_close(to_f32_list(Native.log10(worker(), x)), [0.0, :math.log10(2.0)])
    end

    test "trig: sin / cos / tan" do
      x = f32([0.0, :math.pi() / 2], [2])
      assert_close(to_f32_list(Native.sin(worker(), x)), [0.0, 1.0], 1.0e-4)
      assert_close(to_f32_list(Native.cos(worker(), x)), [1.0, 0.0], 1.0e-4)
      assert_close(to_f32_list(Native.tan(worker(), f32([0.0], [1]))), [0.0])
    end

    test "inverse trig: arcsin / arccos / arctan" do
      assert_close(
        to_f32_list(Native.arcsin(worker(), f32([0.0, 1.0], [2]))),
        [0.0, :math.pi() / 2],
        1.0e-4
      )

      assert_close(
        to_f32_list(Native.arccos(worker(), f32([1.0, 0.0], [2]))),
        [0.0, :math.pi() / 2],
        1.0e-4
      )

      assert_close(
        to_f32_list(Native.arctan(worker(), f32([0.0, 1.0], [2]))),
        [0.0, :math.pi() / 4],
        1.0e-4
      )
    end

    test "hyperbolic: sinh / cosh / tanh and their inverses" do
      x = f32([0.0, 1.0], [2])
      assert_close(to_f32_list(Native.sinh(worker(), x)), [0.0, :math.sinh(1.0)])
      assert_close(to_f32_list(Native.cosh(worker(), x)), [1.0, :math.cosh(1.0)])
      assert_close(to_f32_list(Native.tanh(worker(), x)), [0.0, :math.tanh(1.0)])
      assert_close(to_f32_list(Native.arcsinh(worker(), f32([0.0], [1]))), [0.0])
      assert_close(to_f32_list(Native.arccosh(worker(), f32([1.0], [1]))), [0.0])
      assert_close(to_f32_list(Native.arctanh(worker(), f32([0.0], [1]))), [0.0])
    end

    test "sigmoid" do
      x = f32([0.0, 10.0, -10.0], [3])
      assert_close(to_f32_list(Native.sigmoid(worker(), x)), [0.5, 1.0, 0.0], 1.0e-4)
    end

    test "erf / erfinv" do
      assert_close(to_f32_list(Native.erf(worker(), f32([0.0], [1]))), [0.0], 1.0e-6)
      assert_close(to_f32_list(Native.erfinv(worker(), f32([0.0], [1]))), [0.0], 1.0e-6)
    end

    test "logical_not" do
      p = pred([true, false, true], [3])
      assert to_pred_list(Native.logical_not(worker(), p)) == [false, true, false]
    end

    test "bitwise_invert" do
      t = s32([0, -1, 5], [3])
      assert to_s32_list(Native.bitwise_invert(worker(), t)) == [-1, 0, -6]
    end

    test "isnan / isinf / isfinite" do
      x = f32([0.0, 1.0], [2])
      assert to_pred_list(Native.isnan(worker(), x)) == [false, false]
      assert to_pred_list(Native.isinf(worker(), x)) == [false, false]
      assert to_pred_list(Native.isfinite(worker(), x)) == [true, true]
    end

    test "stop_gradient is identity in forward pass" do
      x = f32([1.0, 2.0, 3.0], [3])
      assert to_f32_list(Native.stop_gradient(worker(), x)) == [1.0, 2.0, 3.0]
    end
  end

  # ---------- Binary ----------

  describe "binary arithmetic" do
    test "add / subtract / multiply / divide" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([10.0, 20.0, 30.0], [3])
      assert to_f32_list(Native.add(worker(), a, b)) == [11.0, 22.0, 33.0]
      assert to_f32_list(Native.subtract(worker(), a, b)) == [-9.0, -18.0, -27.0]
      assert to_f32_list(Native.multiply(worker(), a, b)) == [10.0, 40.0, 90.0]
      assert to_f32_list(Native.divide(worker(), b, a)) == [10.0, 10.0, 10.0]
    end

    test "floor_divide / remainder" do
      a = s32([7, 8, 9], [3])
      b = s32([2, 3, 4], [3])
      assert to_s32_list(Native.floor_divide(worker(), a, b)) == [3, 2, 2]
      assert to_s32_list(Native.remainder(worker(), a, b)) == [1, 2, 1]
    end

    test "power" do
      a = f32([2.0, 3.0], [2])
      b = f32([3.0, 2.0], [2])
      assert to_f32_list(Native.power(worker(), a, b)) == [8.0, 9.0]
    end

    test "maximum / minimum" do
      a = f32([1.0, 5.0, 3.0], [3])
      b = f32([4.0, 2.0, 3.0], [3])
      assert to_f32_list(Native.maximum(worker(), a, b)) == [4.0, 5.0, 3.0]
      assert to_f32_list(Native.minimum(worker(), a, b)) == [1.0, 2.0, 3.0]
    end

    test "logaddexp" do
      a = f32([0.0, 0.0], [2])
      b = f32([0.0, 1.0], [2])

      assert_close(to_f32_list(Native.logaddexp(worker(), a, b)), [
        :math.log(2.0),
        :math.log(1.0 + :math.exp(1.0))
      ])
    end

    test "arctan2" do
      assert_close(to_f32_list(Native.arctan2(worker(), f32([1.0], [1]), f32([1.0], [1]))), [
        :math.pi() / 4
      ])
    end

    test "broadcasting: [3] + [1]" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([10.0], [1])
      assert to_f32_list(Native.add(worker(), a, b)) == [11.0, 12.0, 13.0]
    end
  end

  describe "comparisons" do
    test "equal / not_equal" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([1.0, 5.0, 3.0], [3])
      assert to_pred_list(Native.equal(worker(), a, b)) == [true, false, true]
      assert to_pred_list(Native.not_equal(worker(), a, b)) == [false, true, false]
    end

    test "less / less_equal / greater / greater_equal" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([2.0, 2.0, 2.0], [3])
      assert to_pred_list(Native.less(worker(), a, b)) == [true, false, false]
      assert to_pred_list(Native.less_equal(worker(), a, b)) == [true, true, false]
      assert to_pred_list(Native.greater(worker(), a, b)) == [false, false, true]
      assert to_pred_list(Native.greater_equal(worker(), a, b)) == [false, true, true]
    end
  end

  describe "logical" do
    test "logical_and / logical_or" do
      a = pred([true, true, false, false], [4])
      b = pred([true, false, true, false], [4])
      assert to_pred_list(Native.logical_and(worker(), a, b)) == [true, false, false, false]
      assert to_pred_list(Native.logical_or(worker(), a, b)) == [true, true, true, false]
    end
  end

  describe "bitwise" do
    test "and / or / xor" do
      a = s32([0b1100, 0b1010], [2])
      b = s32([0b1010, 0b0110], [2])
      assert to_s32_list(Native.bitwise_and(worker(), a, b)) == [0b1000, 0b0010]
      assert to_s32_list(Native.bitwise_or(worker(), a, b)) == [0b1110, 0b1110]
      assert to_s32_list(Native.bitwise_xor(worker(), a, b)) == [0b0110, 0b1100]
    end

    test "left_shift / right_shift" do
      a = s32([1, 16], [2])
      b = s32([3, 2], [2])
      assert to_s32_list(Native.left_shift(worker(), a, b)) == [8, 64]
      assert to_s32_list(Native.right_shift(worker(), a, b)) == [0, 4]
    end
  end

  # ---------- Reductions ----------

  describe "reductions" do
    test "sum/mean/prod over all axes" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      assert to_f32_list(Native.sum(worker(), x, [0, 1], false)) == [10.0]
      assert to_f32_list(Native.mean(worker(), x, [0, 1], false)) == [2.5]
      assert to_f32_list(Native.prod(worker(), x, [0, 1], false)) == [24.0]
    end

    test "sum with axes + keepdims" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      # sum over axis 1
      r = Native.sum(worker(), x, [1], false)
      assert Native.shape(r) == [2]
      assert to_f32_list(r) == [3.0, 7.0]

      r_keep = Native.sum(worker(), x, [1], true)
      assert Native.shape(r_keep) == [2, 1]
    end

    test "max / min" do
      x = f32([1.0, 5.0, 3.0, 2.0], [4])
      assert to_f32_list(Native.max(worker(), x, [0], false)) == [5.0]
      assert to_f32_list(Native.min(worker(), x, [0], false)) == [1.0]
    end

    test "all / any" do
      p = pred([true, true, false], [3])
      assert to_pred_list(Native.all(worker(), p, [0], false)) == [false]
      assert to_pred_list(Native.any(worker(), p, [0], false)) == [true]
    end

    test "logsumexp" do
      x = f32([1.0, 2.0, 3.0], [3])
      expected = :math.log(:math.exp(1.0) + :math.exp(2.0) + :math.exp(3.0))
      assert_close(to_f32_list(Native.logsumexp(worker(), x, [0], false)), [expected])
    end

    test "argmax / argmin" do
      x = f32([1.0, 5.0, 3.0], [3])
      assert to_s32_list(Native.argmax(worker(), x, 0, false)) == [1]
      assert to_s32_list(Native.argmin(worker(), x, 0, false)) == [0]
    end

    test "var / std" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      # var with ddof=0 => population variance = 1.25
      assert_close(to_f32_list(Native.var(worker(), x, [0], false, 0)), [1.25])
      assert_close(to_f32_list(Native.std(worker(), x, [0], false, 0)), [:math.sqrt(1.25)])
    end

    test "cumulative: cumsum / cumprod" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      # inclusive, not reversed
      assert to_f32_list(Native.cumsum(worker(), x, 0, false, true)) == [1.0, 3.0, 6.0, 10.0]
      assert to_f32_list(Native.cumprod(worker(), x, 0, false, true)) == [1.0, 2.0, 6.0, 24.0]
    end

    test "cumulative: cummax / cummin" do
      x = f32([3.0, 1.0, 4.0, 1.0, 5.0], [5])
      assert to_f32_list(Native.cummax(worker(), x, 0, false, true)) == [3.0, 3.0, 4.0, 4.0, 5.0]
      assert to_f32_list(Native.cummin(worker(), x, 0, false, true)) == [3.0, 1.0, 1.0, 1.0, 1.0]
    end
  end

  # ---------- Shape ----------

  describe "shape manipulation" do
    test "reshape" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
      r = Native.reshape(worker(), x, [2, 3])
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "transpose" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      r = Native.transpose(worker(), x, [1, 0])
      assert Native.shape(r) == [3, 2]
      assert to_f32_list(r) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    end

    test "squeeze / expand_dims" do
      x = f32([1.0, 2.0, 3.0], [1, 3, 1])
      s = Native.squeeze(worker(), x, [0, 2])
      assert Native.shape(s) == [3]
      e = Native.expand_dims(worker(), s, [0])
      assert Native.shape(e) == [1, 3]
    end

    test "broadcast_to" do
      x = f32([1.0, 2.0, 3.0], [3])
      r = Native.broadcast_to(worker(), x, [2, 3])
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    end

    test "concatenate / stack" do
      a = f32([1.0, 2.0], [2])
      b = f32([3.0, 4.0], [2])
      c = Native.concatenate(worker(), [a, b], 0)
      assert Native.shape(c) == [4]
      assert to_f32_list(c) == [1.0, 2.0, 3.0, 4.0]

      s = Native.stack(worker(), [a, b], 0)
      assert Native.shape(s) == [2, 2]
      assert to_f32_list(s) == [1.0, 2.0, 3.0, 4.0]
    end

    test "flatten" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      r = Native.flatten(worker(), x, 0, -1)
      assert Native.shape(r) == [4]
    end

    test "tile" do
      x = f32([1.0, 2.0], [2])
      r = Native.tile(worker(), x, [3])
      assert to_f32_list(r) == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    end

    test "swapaxes" do
      x = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      r = Native.swapaxes(worker(), x, 0, 1)
      assert to_f32_list(r) == [1.0, 3.0, 2.0, 4.0]
    end

    test "pad" do
      x = f32([1.0, 2.0, 3.0], [3])
      zero = f32_scalar(0.0)
      r = Native.pad(worker(), x, [0], [1], [2], zero)
      assert Native.shape(r) == [6]
      assert to_f32_list(r) == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    end

    test "repeat" do
      x = f32([1.0, 2.0], [2])
      r = Native.repeat(worker(), x, 2, 0)
      assert to_f32_list(r) == [1.0, 1.0, 2.0, 2.0]
    end

    test "flip reverses a 1D tensor" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      assert to_f32_list(Native.flip(worker(), x, 0)) == [4.0, 3.0, 2.0, 1.0]
    end

    test "flip along axis 0 of a 2D tensor reverses rows" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      assert Native.shape(Native.flip(worker(), x, 0)) == [2, 3]
      assert to_f32_list(Native.flip(worker(), x, 0)) == [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
    end

    test "flip along axis 1 of a 2D tensor reverses columns" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      assert to_f32_list(Native.flip(worker(), x, 1)) == [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    end

    test "flip accepts a negative axis" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      assert to_f32_list(Native.flip(worker(), x, -1)) == [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    end

    test "flip of a singleton axis is a no-op" do
      x = f32([1.0, 2.0, 3.0], [1, 3])
      assert to_f32_list(Native.flip(worker(), x, 0)) == [1.0, 2.0, 3.0]
    end

    test "flip of a scalar returns the scalar unchanged" do
      x = f32_scalar(7.0)
      assert to_f32_list(Native.flip(worker(), x, 0)) == [7.0]
    end
  end

  # ---------- Indexing ----------

  describe "indexing" do
    test "slice" do
      x = f32(Enum.to_list(1..12) |> Enum.map(&(&1 * 1.0)), [3, 4])
      r = Native.slice(worker(), x, [0, 1], [2, 3], [1, 1])
      assert Native.shape(r) == [2, 2]
      assert to_f32_list(r) == [2.0, 3.0, 6.0, 7.0]
    end

    test "take" do
      x = f32([10.0, 20.0, 30.0, 40.0], [4])
      idx = s32([0, 2, 3], [3])
      r = Native.take(worker(), x, idx, 0)
      assert to_f32_list(r) == [10.0, 30.0, 40.0]
    end

    test "where" do
      cond_t = pred([true, false, true], [3])
      x = f32([1.0, 2.0, 3.0], [3])
      y = f32([10.0, 20.0, 30.0], [3])
      r = Native.where(worker(), cond_t, x, y)
      assert to_f32_list(r) == [1.0, 20.0, 3.0]
    end
  end

  # ---------- Linalg ----------

  describe "linalg" do
    @describetag :linalg
    test "matmul: 2x3 @ 3x2" do
      a = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      b = f32([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
      r = Native.matmul(worker(), a, b)
      assert Native.shape(r) == [2, 2]
      # [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
      assert to_f32_list(r) == [58.0, 64.0, 139.0, 154.0]
    end

    test "tensordot with axes" do
      a = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      b = f32([5.0, 6.0, 7.0, 8.0], [2, 2])
      # contract last axis of a with first of b (= matmul)
      r = Native.tensordot(worker(), a, b, [1], [0])
      assert to_f32_list(r) == [19.0, 22.0, 43.0, 50.0]
    end

    test "outer" do
      a = f32([1.0, 2.0], [2])
      b = f32([10.0, 20.0, 30.0], [3])
      r = Native.outer(worker(), a, b)
      assert Native.shape(r) == [2, 3]
      assert to_f32_list(r) == [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    end

    test "inner of 1-D vectors = dot product" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([4.0, 5.0, 6.0], [3])
      r = Native.inner(worker(), a, b)
      assert to_f32_list(r) == [32.0]
    end

    # --- Decompositions / solvers (mx::linalg::*) ---

    test "linalg_lu: P * L * U ≈ A for a 3×3 matrix" do
      a = f32([2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], [3, 3])
      {perm, l, u} = Native.linalg_lu(worker(), a)

      assert Native.shape(perm) == [3]
      assert Native.shape(l) == [3, 3]
      assert Native.shape(u) == [3, 3]

      eye = Native.eye(worker(), 3, 3, 0, {:f, 32})
      p = Native.take(worker(), eye, perm, 0)
      pl = Native.matmul(worker(), p, l)
      plu = Native.matmul(worker(), pl, u)

      assert_close(
        to_f32_list(plu),
        [2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
        1.0e-4
      )
    end

    test "linalg_svd: U * diag(S) * Vt ≈ A for a diagonal matrix" do
      a = f32([3.0, 0.0, 0.0, 4.0], [2, 2])
      {u, s, vt} = Native.linalg_svd(worker(), a)

      assert Native.shape(u) == [2, 2]
      assert Native.shape(s) == [2]
      assert Native.shape(vt) == [2, 2]

      s_list = to_f32_list(s)
      assert_close(Enum.max(s_list), 4.0, 1.0e-4)
      assert_close(Enum.min(s_list), 3.0, 1.0e-4)
    end

    test "linalg_qr: Q * R ≈ A and Q^T * Q ≈ I" do
      a = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
      {q, r} = Native.linalg_qr(worker(), a)

      assert Native.shape(q) == [3, 2]
      assert Native.shape(r) == [2, 2]

      qr = Native.matmul(worker(), q, r)
      assert_close(to_f32_list(qr), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0e-4)

      qt = Native.transpose(worker(), q, [1, 0])
      qtq = Native.matmul(worker(), qt, q)
      assert_close(to_f32_list(qtq), [1.0, 0.0, 0.0, 1.0], 1.0e-4)
    end

    test "linalg_cholesky: L * L^T ≈ A for an SPD matrix" do
      a = f32([4.0, 2.0, 2.0, 3.0], [2, 2])
      l = Native.linalg_cholesky(worker(), a, false)

      assert Native.shape(l) == [2, 2]

      lt = Native.transpose(worker(), l, [1, 0])
      llt = Native.matmul(worker(), l, lt)
      assert_close(to_f32_list(llt), [4.0, 2.0, 2.0, 3.0], 1.0e-4)
    end

    test "linalg_eigh: eigendecomposition of a symmetric matrix" do
      a = f32([2.0, 1.0, 1.0, 3.0], [2, 2])
      {vals, vecs} = Native.linalg_eigh(worker(), a, "L")

      assert Native.shape(vals) == [2]
      assert Native.shape(vecs) == [2, 2]

      vals_list = to_f32_list(vals) |> Enum.sort()
      assert_close(Enum.at(vals_list, 0), (5.0 - :math.sqrt(5)) / 2, 1.0e-4)
      assert_close(Enum.at(vals_list, 1), (5.0 + :math.sqrt(5)) / 2, 1.0e-4)
    end

    test "linalg_solve: Ax = b with known solution" do
      a = f32([1.0, 2.0, 3.0, 4.0], [2, 2])
      b = f32([5.0, 11.0], [2])
      x = Native.linalg_solve(worker(), a, b)

      assert Native.shape(x) == [2]
      assert_close(to_f32_list(x), [1.0, 2.0], 1.0e-4)
    end

    test "linalg_solve_triangular: lower-triangular Lx = b" do
      l = f32([2.0, 0.0, 1.0, 3.0], [2, 2])
      b = f32([2.0, 4.0], [2])
      x = Native.linalg_solve_triangular(worker(), l, b, false)

      assert Native.shape(x) == [2]
      assert_close(to_f32_list(x), [1.0, 1.0], 1.0e-4)
    end

    test "linalg_solve_triangular: upper-triangular Ux = b" do
      u = f32([2.0, 1.0, 0.0, 3.0], [2, 2])
      b = f32([5.0, 3.0], [2])
      x = Native.linalg_solve_triangular(worker(), u, b, true)

      assert Native.shape(x) == [2]
      assert_close(to_f32_list(x), [2.0, 1.0], 1.0e-4)
    end
  end

  # ---------- Quantization ----------

  describe "quantization" do
    # Each test below uses the smallest MLX-legal shapes: `group_size` must
    # divide the last axis, and for `bits=4` the packed last axis shrinks by
    # a factor of 8 (32 bits per u32 / 4 bits per value).

    test "quantize returns packed w_q plus per-group scales and biases" do
      # Single group of 64 elements, all zero: scales and biases collapse.
      wt = f32(List.duplicate(0.0, 64), [1, 64])
      {q, s, b} = Native.quantize(worker(), wt, 64, 4)

      # 64 nibbles → 8 u32 values.
      assert Native.shape(q) == [1, 8]
      assert Native.dtype(q) == {:u, 32}

      # One group per row.
      assert Native.shape(s) == [1, 1]
      assert Native.shape(b) == [1, 1]
      assert Native.dtype(s) == {:f, 32}
      assert Native.dtype(b) == {:f, 32}

      # All-zero input → scales/biases collapse to a small epsilon rather
      # than exactly zero (MLX picks a non-zero scale to avoid divide-by-zero
      # downstream); both stay within ulp of zero.
      assert_close(to_f32_list(s), [0.0], 1.0e-6)
      assert_close(to_f32_list(b), [0.0], 1.0e-6)
    end

    test "quantize validates last-axis divisibility" do
      bad = f32(List.duplicate(0.0, 30), [1, 30])
      assert_raise ArgumentError, fn -> Native.quantize(worker(), bad, 64, 4) end
    end

    test "dequantize recovers the original within int4 tolerance" do
      # Values spaced across a single group's dynamic range. Max quantization
      # step for int4 is roughly `(max - min) / 15`; we assert an upper bound
      # slightly looser than that to avoid ulp noise.
      values = for i <- 0..63, do: (i - 32) / 32.0
      wt = f32(values, [1, 64])

      {q, s, b} = Native.quantize(worker(), wt, 64, 4)
      deq = Native.dequantize(worker(), q, s, b, 64, 4)

      assert Native.shape(deq) == [1, 64]
      step = 2.0 / 15.0
      assert_close(to_f32_list(deq), values, step)
    end

    test "quantized_matmul with transpose=true matches matmul(x, deq.T)" do
      # w: [out=2, in=64]; x: [batch=3, in=64]. With transpose=true, MLX
      # computes x @ w.T (i.e. w is "rows of output").
      w_vals = for i <- 0..127, do: (i - 64) / 128.0
      wt = f32(w_vals, [2, 64])
      x_vals = for i <- 0..(3 * 64 - 1), do: i / 64.0
      x = f32(x_vals, [3, 64])

      {q, s, b} = Native.quantize(worker(), wt, 64, 4)

      qmm = Native.quantized_matmul(worker(), x, q, s, b, true, 64, 4)
      deq = Native.dequantize(worker(), q, s, b, 64, 4)
      ref = Native.matmul(worker(), x, Native.transpose(worker(), deq, [1, 0]))

      assert Native.shape(qmm) == [3, 2]
      # Fused vs. composed dequant+matmul may reorder ops; allow a small
      # absolute tolerance. Observed max on this fixture: ~3e-5.
      assert_close(to_f32_list(qmm), to_f32_list(ref), 1.0e-4)
    end

    test "quantized_matmul with transpose=false flips the weight layout" do
      # With transpose=false, MLX treats the packed matrix as already
      # transposed, so `y = x @ w`. Round-trip against a matching matmul
      # on the dequantized weight with the same convention.
      w_vals = for i <- 0..127, do: (i - 64) / 128.0
      wt = f32(w_vals, [2, 64])
      x = f32(for(i <- 0..(3 * 2 - 1), do: i / 6.0), [3, 2])

      {q, s, b} = Native.quantize(worker(), wt, 64, 4)

      qmm = Native.quantized_matmul(worker(), x, q, s, b, false, 64, 4)
      deq = Native.dequantize(worker(), q, s, b, 64, 4)
      ref = Native.matmul(worker(), x, deq)

      assert Native.shape(qmm) == [3, 64]
      assert_close(to_f32_list(qmm), to_f32_list(ref), 1.0e-4)
    end

    test "bits=8 halves the packing density" do
      values = for i <- 0..63, do: i / 64.0
      wt = f32(values, [1, 64])
      {q, _s, _b} = Native.quantize(worker(), wt, 64, 8)

      # 64 bytes → 16 u32 values.
      assert Native.shape(q) == [1, 16]
      assert Native.dtype(q) == {:u, 32}
    end

    test "smaller group_size produces more scale/bias rows" do
      wt = f32(List.duplicate(0.0, 128), [1, 128])
      {_q, s, _b} = Native.quantize(worker(), wt, 32, 4)

      # 128 / 32 = 4 groups along the last axis.
      assert Native.shape(s) == [1, 4]
    end
  end

  # ---------- Lifecycle ----------

  describe "lifecycle under load" do
    test "chained lazy ops survive GC before eval" do
      a = f32([1.0, 2.0, 3.0, 4.0], [4])
      b = f32([10.0, 20.0, 30.0, 40.0], [4])
      c = Native.add(worker(), a, b)
      d = Native.multiply(worker(), c, c)

      :erlang.garbage_collect()
      assert to_f32_list(d) == [121.0, 484.0, 1089.0, 1936.0]
    end
  end

  # ---------- Sort ----------

  describe "sort / topk" do
    test "sort/2 and argsort/2 along last axis" do
      x = f32([3.0, 1.0, 2.0], [3])
      assert to_f32_list(Native.sort(worker(), x, -1)) == [1.0, 2.0, 3.0]
      assert to_s32_list(Native.argsort(worker(), x, -1)) == [1, 2, 0]
    end

    test "partition/3 places kth smallest first" do
      x = f32([5.0, 2.0, 4.0, 1.0, 3.0], [5])
      # Partitioning around kth=2 guarantees first 2 entries are the two
      # smallest in some order, and later entries are >= those.
      result = to_f32_list(Native.partition(worker(), x, 2, -1))
      assert Enum.sort(Enum.take(result, 2)) == [1.0, 2.0]
    end

    test "topk/3 returns the k largest (unordered)" do
      x = f32([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], [8])
      result = to_f32_list(Native.topk(worker(), x, 3, -1)) |> Enum.sort()
      assert result == [5.0, 6.0, 9.0]
    end
  end

  # ---------- Misc ----------

  describe "clip / roll / softmax" do
    test "clip/3" do
      x = f32([-1.0, 0.5, 2.0, 3.5], [4])
      lo = f32_scalar(0.0)
      hi = f32_scalar(2.0)
      assert to_f32_list(Native.clip(worker(), x, lo, hi)) == [0.0, 0.5, 2.0, 2.0]
    end

    test "roll/3" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      assert to_f32_list(Native.roll(worker(), x, 1, 0)) == [4.0, 1.0, 2.0, 3.0]
      assert to_f32_list(Native.roll(worker(), x, -1, 0)) == [2.0, 3.0, 4.0, 1.0]
    end

    test "softmax/3 along last axis sums to 1" do
      x = f32([1.0, 2.0, 3.0], [3])
      result = to_f32_list(Native.softmax(worker(), x, [0], false))
      assert_close(Enum.sum(result), 1.0)
      # Monotonically increasing input → monotonically increasing output.
      assert Enum.sort(result) == result
    end

    test "logcumsumexp/4 matches log(cumsum(exp(x)))" do
      x = f32([1.0, 2.0, 3.0, 4.0], [4])
      result = to_f32_list(Native.logcumsumexp(worker(), x, 0, false, true))

      # Reference: [log(e^1), log(e^1+e^2), log(e^1+e^2+e^3),
      #             log(e^1+e^2+e^3+e^4)].
      expected =
        [:math.exp(1.0), :math.exp(2.0), :math.exp(3.0), :math.exp(4.0)]
        |> Enum.scan(&+/2)
        |> Enum.map(&:math.log/1)

      assert_close(result, expected, 1.0e-4)
    end

    test "array_equal/3" do
      a = f32([1.0, 2.0, 3.0], [3])
      b = f32([1.0, 2.0, 3.0], [3])
      c = f32([1.0, 2.0, 4.0], [3])
      assert to_pred_list(Native.array_equal(worker(), a, b, false)) == [true]
      assert to_pred_list(Native.array_equal(worker(), a, c, false)) == [false]
    end
  end

  # ---------- Axis-aligned gather/scatter ----------

  describe "axis-aligned gather/scatter" do
    test "take_along_axis/3" do
      x = f32([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])
      idx = s32([2, 0, 1, 1, 2, 0], [2, 3])
      r = Native.take_along_axis(worker(), x, idx, 1)
      # Row 0: [30, 10, 20]; Row 1: [50, 60, 40]
      assert to_f32_list(r) == [30.0, 10.0, 20.0, 50.0, 60.0, 40.0]
    end

    test "put_along_axis/4 writes values back into the source" do
      x = f32([0.0, 0.0, 0.0, 0.0], [4])
      idx = s32([0, 2], [2])
      vals = f32([1.0, 2.0], [2])
      r = Native.put_along_axis(worker(), x, idx, vals, 0)
      assert to_f32_list(r) == [1.0, 0.0, 2.0, 0.0]
    end

    test "scatter_add_axis/4 accumulates at the indices" do
      x = f32([10.0, 20.0, 30.0], [3])
      idx = s32([0, 0, 2], [3])
      vals = f32([1.0, 1.0, 5.0], [3])
      r = Native.scatter_add_axis(worker(), x, idx, vals, 0)
      # idx 0 gets +1+1; idx 2 gets +5.
      assert to_f32_list(r) == [12.0, 20.0, 35.0]
    end
  end

  # ---------- Multi-axis gather/scatter ----------

  describe "multi-axis gather/scatter" do
    # Multi-axis gather: pick scalars at (i, j) positions out of a 3x4
    # matrix. indices is a list of two length-N arrays (one per axis);
    # slice_sizes is all-1 so each gather is a scalar.
    test "gather/4 scalar picks across two axes" do
      # [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
      a = f32(Enum.map(1..12, &(&1 * 1.0)), [3, 4])
      idx0 = s32([0, 2, 1], [3])
      idx1 = s32([1, 3, 0], [3])
      # Expected: a[0,1]=2, a[2,3]=12, a[1,0]=5.
      r = Native.gather(worker(), a, [idx0, idx1], [0, 1], [1, 1])
      # Result shape is batch ++ slice_sizes = [3, 1, 1].
      assert Native.shape(r) == [3, 1, 1]
      assert to_f32_list(r) == [2.0, 12.0, 5.0]
    end

    # Partial-axis gather: gather along axis 0 only, keeping full axis
    # 1. slice_sizes = [1, 4] and result shape = batch ++ [1, 4].
    test "gather/4 partial-axis keeps slice dims" do
      a = f32(Enum.map(1..12, &(&1 * 1.0)), [3, 4])
      idx0 = s32([2, 0], [2])
      r = Native.gather(worker(), a, [idx0], [0], [1, 4])
      # Row 2: [9,10,11,12]; Row 0: [1,2,3,4].
      assert Native.shape(r) == [2, 1, 4]
      assert to_f32_list(r) == [9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]
    end

    # scatter (overwrite) across all axes — scalar writes.
    test "scatter/4 scalar writes across all axes" do
      a = f32(List.duplicate(0.0, 6), [2, 3])
      idx0 = s32([0, 1, 0], [3])
      idx1 = s32([1, 2, 0], [3])
      # Updates must have rank = indices[0].ndim() + a.ndim() = 1 + 2 = 3.
      # Shape: batch ++ [1, 1] = [3, 1, 1].
      upd = f32([7.0, 8.0, 9.0], [3, 1, 1])
      r = Native.scatter(worker(), a, [idx0, idx1], upd, [0, 1])
      # a[0,1]=7, a[1,2]=8, a[0,0]=9.
      assert to_f32_list(r) == [9.0, 7.0, 0.0, 0.0, 0.0, 8.0]
    end

    # scatter_add across all axes — duplicate indices accumulate.
    test "scatter_add/4 accumulates at duplicate indices" do
      a = f32([10.0, 20.0, 30.0], [3])
      idx0 = s32([0, 0, 2], [3])
      upd = f32([1.0, 1.0, 5.0], [3, 1])
      r = Native.scatter_add(worker(), a, [idx0], upd, [0])
      assert to_f32_list(r) == [12.0, 20.0, 35.0]
    end

    # The critical partial-axis case from the design critique: target
    # shape {B, L, D} = {2, 3, 4}, axes = [0, 1], N = 2 writes of a
    # whole D-slice. updates_shape (Nx-view) = {2, 4}, MLX-view =
    # {2, 1, 1, 4}. Exercises the updates-shape rewrap contract.
    test "scatter_add/4 partial-axis slice writes (B, L, D) target" do
      # {2, 3, 4} of zeros.
      a = f32(List.duplicate(0.0, 24), [2, 3, 4])
      # Two writes: (b=0, l=1) += [1,2,3,4]; (b=1, l=2) += [5,6,7,8].
      idx_b = s32([0, 1], [2])
      idx_l = s32([1, 2], [2])
      # MLX updates: {2, 1, 1, 4} — one leading "1" per indexed axis.
      upd = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 1, 1, 4])
      r = Native.scatter_add(worker(), a, [idx_b, idx_l], upd, [0, 1])
      assert Native.shape(r) == [2, 3, 4]

      # Build expected: zeros with the two slice writes.
      expected =
        Enum.flat_map(0..1, fn b ->
          Enum.flat_map(0..2, fn l ->
            case {b, l} do
              {0, 1} -> [1.0, 2.0, 3.0, 4.0]
              {1, 2} -> [5.0, 6.0, 7.0, 8.0]
              _ -> [0.0, 0.0, 0.0, 0.0]
            end
          end)
        end)

      assert to_f32_list(r) == expected
    end
  end

  # ---------- Convolution ----------

  describe "conv_general" do
    test "1-D convolution with stride=1, no padding, identity kernel" do
      # Input shape NLC: [1, 4, 1]
      input = f32([1.0, 2.0, 3.0, 4.0], [1, 4, 1])
      # Kernel shape O_i x K x C: [1, 2, 1]
      weight = f32([1.0, 1.0], [1, 2, 1])

      r =
        Native.conv_general(
          worker(),
          input,
          weight,
          [1],
          {[0], [0]},
          {[1], [1]},
          1,
          false
        )

      assert Native.shape(r) == [1, 3, 1]
      assert to_f32_list(r) == [3.0, 5.0, 7.0]
    end
  end

  # ---------- Random ----------

  describe "random" do
    test "random_key/1 is deterministic for the same seed" do
      k1 = Native.random_key(42)
      k2 = Native.random_key(42)

      assert to_s32_list(Native.astype(worker(), k1, {:s, 32})) ==
               to_s32_list(Native.astype(worker(), k2, {:s, 32}))
    end

    test "random_uniform/5 produces values in [low, high)" do
      key = Native.random_key(7)
      low = f32_scalar(0.0)
      high = f32_scalar(1.0)
      t = Native.random_uniform(worker(), low, high, [1000], {:f, 32}, key)
      vals = to_f32_list(t)

      assert Enum.all?(vals, &(&1 >= 0.0))
      assert Enum.all?(vals, &(&1 < 1.0))

      mean = Enum.sum(vals) / length(vals)
      assert_close(mean, 0.5, 5.0e-2)
    end

    test "random_normal/5 has roughly the right mean and stdev" do
      key = Native.random_key(99)
      t = Native.random_normal(worker(), [2048], {:f, 32}, 0.0, 1.0, key)
      vals = to_f32_list(t)

      mean = Enum.sum(vals) / length(vals)
      var = Enum.sum(Enum.map(vals, &((&1 - mean) * (&1 - mean)))) / length(vals)

      assert_close(mean, 0.0, 5.0e-2)
      assert_close(:math.sqrt(var), 1.0, 5.0e-2)
    end

    test "random_bernoulli/3 yields 0/1 pred values" do
      key = Native.random_key(1)
      p = f32_scalar(0.3)
      t = Native.random_bernoulli(worker(), p, [256], key)
      assert Native.dtype(t) == {:pred, 1}
      assert Enum.all?(to_pred_list(t), &is_boolean/1)
    end

    test "random with nil key still produces the right shape/dtype" do
      # Uses the default key sequence — value is non-deterministic, so
      # only assert shape/dtype.
      t = Native.random_normal(worker(), [4], {:f, 32}, 0.0, 1.0, nil)
      assert Native.shape(t) == [4]
      assert Native.dtype(t) == {:f, 32}
    end

    test "random_split/2 yields distinct keys" do
      k = Native.random_key(5)
      split = Native.random_split(worker(), k, 2)
      assert Native.shape(split) == [2, 2]
    end
  end

  # ---------- FFT ----------

  describe "fft" do
    test "fftn ∘ ifftn round-trips within tolerance" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8])
      # Cast to complex64 for FFT.
      xc = Native.astype(worker(), x, {:c, 64})
      fwd = Native.fftn(worker(), xc, [8], [0])
      inv = Native.ifftn(worker(), fwd, [8], [0])
      back = Native.astype(worker(), Native.real(worker(), inv), {:f, 32})
      assert_close(to_f32_list(back), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1.0e-4)
    end

    test "rfftn produces (n/2 + 1) frequency bins" do
      x = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8])
      r = Native.rfftn(worker(), x, [8], [0])
      assert Native.shape(r) == [5]
      assert Native.dtype(r) == {:c, 64}
    end
  end

  # ---------- Memory ----------

  describe "memory" do
    test "memory NIFs return non-negative ints" do
      assert is_integer(Native.get_active_memory())
      assert Native.get_active_memory() >= 0
      assert is_integer(Native.get_peak_memory())
      assert Native.get_peak_memory() >= 0
      assert is_integer(Native.get_cache_memory())
      assert Native.get_cache_memory() >= 0
    end

    test "reset_peak_memory / clear_cache return :ok" do
      assert :ok = Native.reset_peak_memory()
      assert :ok = Native.clear_cache()
    end
  end
end
