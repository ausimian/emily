defmodule Emily.Quantization.QuantizedMatmulTest do
  @moduledoc """
  Property tests for `Emily.Quantization.quantized_matmul/2` — the
  direct-call helper that dispatches to `Native.quantized_matmul` on
  materialized Emily tensors.

  The oracle is `Nx.BinaryBackend` dot on the *dequantized* weight: the
  MLX quantized_matmul kernel is mathematically equivalent to
  dequantize-then-dot (the two only diverge at the accumulation rounding
  level), so the binary-backend ref is what we actually want to compare
  against. Tolerance accounts for int4 quantization error plus a small
  accumulation-order term.
  """

  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Emily.Quantization
  alias Emily.QuantizedWeight

  import Emily.BackendGenerators

  describe "quantized_matmul/2 — transpose=true (fresh-from-dense layout)" do
    property "matches Nx.dot(x, Nx.transpose(to_dense(qw)))" do
      check all(
              batch <- StreamData.integer(1..4),
              out_feat <- StreamData.integer(1..4),
              groups <- StreamData.integer(1..3),
              max_runs: 10
            ) do
        in_feat = groups * 64

        # Keep values modest so int4 error stays in a tight band.
        w =
          Nx.iota({out_feat, in_feat}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(out_feat * in_feat / 2)
          |> Nx.subtract(1.0)

        x =
          Nx.iota({batch, in_feat}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(in_feat)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w)

        actual = Quantization.quantized_matmul(x, qw)

        # Oracle: dequantize then dot on the BinaryBackend. Using the
        # dequantized value (not the original `w`) removes the
        # quantization-error component from the comparison; what's left is
        # pure floating-point reordering.
        dense = QuantizedWeight.to_dense(qw) |> Nx.backend_transfer(Nx.BinaryBackend)
        x_ref = Nx.backend_transfer(x, Nx.BinaryBackend)
        expected = Nx.dot(x_ref, Nx.transpose(dense))

        assert Nx.shape(actual) == {batch, out_feat}
        assert_close(actual, expected, tol: 1.0e-3)
      end
    end
  end

  describe "quantized_matmul/2 — transpose=false (AWQ-style layout)" do
    test "matches Nx.dot(x, to_dense(qw)) without transpose" do
      w_vals = for i <- 0..127, do: (i - 64) / 128.0

      w =
        Nx.tensor(w_vals, backend: Emily.Backend, type: :f32)
        |> Nx.reshape({2, 64})

      x_vals = for i <- 0..5, do: i / 6.0 - 0.25
      x = Nx.tensor(x_vals, backend: Emily.Backend, type: :f32) |> Nx.reshape({3, 2})

      qw = QuantizedWeight.from_dense(w, transpose: false)

      actual = Quantization.quantized_matmul(x, qw)

      # With transpose=false, MLX computes x @ w directly (no transpose
      # of the packed matrix), so the oracle is `Nx.dot(x, dense)`.
      dense = QuantizedWeight.to_dense(qw) |> Nx.backend_transfer(Nx.BinaryBackend)
      x_ref = Nx.backend_transfer(x, Nx.BinaryBackend)
      expected = Nx.dot(x_ref, dense)

      assert Nx.shape(actual) == {3, 64}
      assert_close(actual, expected, tol: 1.0e-3)
    end
  end

  describe "quantized_matmul/2 — validation" do
    test "raises when input dtype doesn't match scales dtype" do
      w = Nx.iota({2, 64}, backend: Emily.Backend, type: :f32)
      qw = QuantizedWeight.from_dense(w)
      x = Nx.iota({3, 64}, backend: Emily.Backend, type: :f32) |> Nx.as_type(:f16)

      assert_raise ArgumentError, ~r/must match scales dtype/, fn ->
        Quantization.quantized_matmul(x, qw)
      end
    end

    test "transfers a BinaryBackend input onto Emily.Backend" do
      w = Nx.iota({2, 64}, backend: Emily.Backend, type: :f32)
      qw = QuantizedWeight.from_dense(w)
      x = Nx.iota({3, 64}, backend: Nx.BinaryBackend, type: :f32)

      # No error: the helper transfers x before dispatch.
      out = Quantization.quantized_matmul(x, qw)
      assert Nx.shape(out) == {3, 2}
    end
  end
end
