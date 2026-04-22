defmodule Emily.Quantization.DequantizeDefnTest do
  @moduledoc """
  Equality tests for `Emily.Quantization.dequantize_defn/1` against
  `QuantizedWeight.to_dense/1` (the Native path). Covers every
  `bits ∈ {2, 4, 8}` × `group_size ∈ {32, 64, 128}` combo, plus a
  rank-3 case and defn composability. `bits ∈ {3, 6}` are rejected by
  the defn path; negative test below.
  """

  use ExUnit.Case, async: true
  doctest Emily.Quantization

  alias Emily.Quantization
  alias Emily.QuantizedWeight

  import Emily.BackendGenerators, only: [assert_close: 3]

  @bits [2, 4, 8]
  @group_sizes [32, 64, 128]

  describe "dequantize_defn/1 — equality with QuantizedWeight.to_dense/1" do
    for bits <- @bits, group_size <- @group_sizes do
      test "bits=#{bits}, group_size=#{group_size}" do
        bits = unquote(bits)
        group_size = unquote(group_size)

        out_feat = 4
        in_feat = group_size * 3

        # Small-magnitude values so int4 quantization error doesn't
        # swamp the comparison — which is against the *dequantized*
        # oracle, not the original dense, so the quantization error
        # cancels anyway.
        w =
          Nx.iota({out_feat, in_feat}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(out_feat * in_feat / 2)
          |> Nx.subtract(1.0)

        qw = QuantizedWeight.from_dense(w, group_size: group_size, bits: bits)

        actual = Quantization.dequantize_defn(qw)
        expected = QuantizedWeight.to_dense(qw)

        assert Nx.shape(actual) == Nx.shape(expected)
        assert Nx.type(actual) == Nx.type(expected)
        # f32 reconstruction is mathematically identical on both paths;
        # 1e-6 is a tight safety tolerance.
        assert_close(actual, expected, tol: 1.0e-6)
      end
    end

    test "rank ≥ 3 input (batch of matrices)" do
      # QuantizedWeight itself requires rank ≥ 2, but the last-axis
      # quantization scheme means any leading axes are carried through
      # unchanged. Verify that the defn path handles rank-3.
      w =
        Nx.iota({2, 3, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(2 * 3 * 128 / 2)
        |> Nx.subtract(1.0)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4)

      actual = Quantization.dequantize_defn(qw)
      expected = QuantizedWeight.to_dense(qw)

      assert Nx.shape(actual) == {2, 3, 128}
      assert_close(actual, expected, tol: 1.0e-6)
    end
  end

  describe "dequantize_defn/1 — composes inside defn" do
    import Nx.Defn

    defn dequantize_then_scale(qw, factor) do
      Emily.Quantization.dequantize_defn(qw) * factor
    end

    test "runs under Nx.Defn.jit" do
      w =
        Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(128.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4)
      factor = Nx.tensor(2.0, backend: Emily.Backend, type: :f32)

      actual = dequantize_then_scale(qw, factor)
      expected = QuantizedWeight.to_dense(qw) |> Nx.multiply(2.0)

      assert Nx.shape(actual) == {2, 128}
      assert_close(actual, expected, tol: 1.0e-6)
    end
  end

  describe "dequantize_defn/1 — validation" do
    test "raises on bits=3 (cross-u32 packing not supported)" do
      w = Nx.iota({2, 64}, backend: Emily.Backend, type: :f32) |> Nx.divide(128.0)
      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 3)

      assert_raise ArgumentError, ~r/bits=3 uses cross-u32/, fn ->
        Quantization.dequantize_defn(qw)
      end
    end

    test "raises on bits=6 (cross-u32 packing not supported)" do
      w = Nx.iota({2, 64}, backend: Emily.Backend, type: :f32) |> Nx.divide(128.0)
      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 6)

      assert_raise ArgumentError, ~r/bits=6 uses cross-u32/, fn ->
        Quantization.dequantize_defn(qw)
      end
    end

    test "raises on microscaled mode (defn path is affine-only)" do
      w = Nx.iota({2, 128}, backend: Emily.Backend, type: :f32) |> Nx.divide(256.0)
      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)

      assert_raise ArgumentError, ~r/mode=.*mxfp4.*to_dense/, fn ->
        Quantization.dequantize_defn(qw)
      end
    end
  end
end
