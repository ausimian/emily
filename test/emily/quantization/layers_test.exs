defmodule Emily.Quantization.LayersTest do
  @moduledoc """
  Tests for `Emily.Quantization.Layers.quantized_dense/4` — the
  Axon-compatible layer op that backs `Emily.Quantization.Transform`.

  Covers the layer op itself:

    * matches the eager-mode `Emily.Quantization.quantized_matmul/2`
      path for the `transpose=true` (MLX / fresh-from-dense) layout;
    * matches `Nx.dot(x, dense)` for the `transpose=false` (AWQ / Axon
      kernel) layout;
    * bias broadcasts correctly;
    * composes inside a `defn`.

  Full Axon-graph-with-quantized-params integration is covered by
  `Emily.Quantization.TransformTest` (Phase 3) — that path quantizes a
  model state *after* init (mirroring `Axon.Quantization.quantize_model_state/2`),
  which is the realistic Bumblebee-loading pattern and doesn't require
  a defn-traceable `from_dense`.
  """

  use ExUnit.Case, async: true

  alias Emily.Quantization
  alias Emily.Quantization.Layers
  alias Emily.QuantizedWeight

  import Emily.BackendGenerators, only: [assert_close: 3]

  describe "quantized_dense/4 — transpose=true (MLX / PyTorch layout)" do
    test "matches Emily.Quantization.quantized_matmul/2" do
      # Kernel in [out, in] layout — MLX / PyTorch convention. Groups
      # along the last axis (in = reduction axis).
      w =
        Nx.iota({4, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(4 * 128 / 2)
        |> Nx.subtract(1.0)

      x =
        Nx.iota({3, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(128.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4, transpose: true)

      actual = Layers.quantized_dense(x, qw)
      expected = Quantization.quantized_matmul(x, qw)

      assert Nx.shape(actual) == {3, 4}
      # Layers path dequantizes then dots; eager path fuses. Both reduce
      # to the same math; allow MLX's fp accumulation reordering.
      assert_close(actual, expected, tol: 1.0e-3)
    end

    test "bias broadcasts through" do
      w =
        Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(128.0)
        |> Nx.subtract(1.0)

      x =
        Nx.iota({3, 64}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(64.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4, transpose: true)
      b = Nx.tensor([0.1, -0.2, 0.3, -0.4], backend: Emily.Backend, type: :f32)

      actual = Layers.quantized_dense(x, qw, b)
      expected = Quantization.quantized_matmul(x, qw) |> Nx.add(b)

      assert_close(actual, expected, tol: 1.0e-3)
    end
  end

  describe "quantized_dense/4 — transpose=false (AWQ / Axon-kernel layout)" do
    test "matches Nx.dot(x, dense) directly" do
      # Kernel in [in, out] layout but with `out` divisible by the
      # group size — this is how MLX's transpose=false convention
      # works: groups are along the last axis (= out), reduction is
      # along the first axis (= in). This is the convention AWQ
      # checkpoints ship in on HF.
      in_feat = 2
      out_feat = 128

      w =
        Nx.iota({in_feat, out_feat}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(in_feat * out_feat / 2)
        |> Nx.subtract(1.0)

      x =
        Nx.iota({3, in_feat}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(in_feat * 1.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4, transpose: false)

      actual = Layers.quantized_dense(x, qw)
      expected = Quantization.quantized_matmul(x, qw)

      assert Nx.shape(actual) == {3, out_feat}
      assert_close(actual, expected, tol: 1.0e-3)
    end
  end

  describe "quantized_dense_impl — composes inside an outer defn" do
    # Prove the layer's defnp core is Expr-traceable when embedded in a
    # larger defn function (the realistic Bumblebee calling pattern).
    import Nx.Defn

    defn layer_then_scale(x, qw, scale) do
      Layers.quantized_dense(x, qw) * scale
    end

    test "runs end-to-end under Nx.Defn.jit" do
      in_feat = 2
      out_feat = 64

      w =
        Nx.iota({in_feat, out_feat}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(in_feat * out_feat / 2)
        |> Nx.subtract(1.0)

      x =
        Nx.iota({3, in_feat}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(in_feat * 1.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, group_size: 64, bits: 4, transpose: false)
      scale = Nx.tensor(2.0, backend: Emily.Backend, type: :f32)

      actual = layer_then_scale(x, qw, scale)
      expected = Quantization.quantized_matmul(x, qw) |> Nx.multiply(2.0)

      assert Nx.shape(actual) == {3, out_feat}
      assert_close(actual, expected, tol: 1.0e-3)
    end
  end
end
