defmodule Emily.Conformance.Bonsai1BitFullTest do
  @moduledoc """
  1-bit affine quantization against a real checkpoint:
  `prism-ml/Bonsai-1.7B-mlx-1bit`, a Qwen3 model that MLX-LM packed
  at `bits: 1, group_size: 128`.

  Excluded from the default suite because it downloads the ~270 MB
  checkpoint on first run. Run explicitly:

      mix test --only bonsai_full

  Emily has no loader for pre-quantized MLX checkpoints, so this test
  builds `QuantizedWeight`s straight from the safetensors entries
  (`<layer>.weight` packed u32, `.scales` and `.biases` f16) and checks
  that:

    * re-quantizing the dequantized weight reproduces the checkpoint's
      packed bits, scales and biases exactly, so Emily packs 1-bit
      weights the way MLX-LM does;
    * `dequantize_defn/1` matches `to_dense/1` bit for bit;
    * `quantized_matmul/2`, eager and natively compiled, matches a
      dense f32 matmul against the dequantized weight.
  """

  use ExUnit.Case, async: true

  alias Bumblebee.HuggingFace.Hub
  alias Emily.Quantization
  alias Emily.QuantizedWeight

  @moduletag :bonsai_full
  @moduletag timeout: 600_000

  @repo "prism-ml/Bonsai-1.7B-mlx-1bit"
  @revision "fac480d016cf265838ad087f9f3f2abac4a2c1a5"

  # First, middle and last layers, covering every projection shape:
  # in 2048 → out 1024/2048/6144, and down_proj's in 6144 → out 2048.
  @layers for i <- [0, 13, 27],
              proj <- ~w(self_attn.q_proj self_attn.k_proj self_attn.o_proj
                           mlp.gate_proj mlp.down_proj),
              do: "model.layers.#{i}.#{proj}"

  setup_all do
    url = Hub.file_url(@repo, "model.safetensors", @revision)

    {:ok, path} =
      Hub.cached_download(url,
        cache_scope: String.replace(@repo, "/", "--")
      )

    %{tensors: Safetensors.read!(path, lazy: true)}
  end

  for prefix <- @layers ++ ["model.embed_tokens"] do
    test "#{prefix}", %{tensors: tensors} do
      qw = quantized_weight(tensors, unquote(prefix))
      dense = QuantizedWeight.to_dense(qw)

      repacked = QuantizedWeight.from_dense(dense, bits: 1, group_size: 128)
      assert_equal(repacked.value, qw.value)
      assert_equal(repacked.scales, qw.scales)
      assert_equal(repacked.biases, qw.biases)

      assert_equal(Quantization.dequantize_defn(qw), dense)

      in_features = elem(Nx.shape(qw.scales), 1) * 128
      {x, _key} = Nx.Random.normal(Nx.Random.key(42), shape: {4, in_features}, type: :f16)
      x = Nx.backend_transfer(x, Emily.Backend)

      expected = Nx.dot(Nx.as_type(x, :f32), [1], Nx.as_type(dense, :f32), [1])

      # f16 activations and accumulation against an f32 reference; the
      # observed worst case across these weights is ~3.5e-4.
      assert_rel_close(Quantization.quantized_matmul(x, qw), expected, 1.0e-3)
    end
  end

  test "quantized_matmul_defn compiles natively on a real 1-bit weight", %{tensors: tensors} do
    qw = quantized_weight(tensors, "model.layers.0.mlp.down_proj")
    {x, _key} = Nx.Random.normal(Nx.Random.key(7), shape: {2, 6144}, type: :f16)
    x = Nx.backend_transfer(x, Emily.Backend)

    native =
      Nx.Defn.jit(&Quantization.quantized_matmul_defn/2,
        compiler: Emily.Compiler,
        native: true,
        native_fallback: :raise
      ).(x, qw)

    assert Nx.shape(native) == {2, 2048}
    assert_equal(native, Quantization.quantized_matmul(x, qw))
  end

  defp quantized_weight(tensors, prefix) do
    load = fn suffix ->
      tensors
      |> Map.fetch!(prefix <> suffix)
      |> Nx.to_tensor()
      |> Nx.backend_transfer(Emily.Backend)
    end

    %QuantizedWeight{
      value: load.(".weight"),
      scales: load.(".scales"),
      biases: load.(".biases"),
      group_size: 128,
      bits: 1,
      transpose: true,
      mode: "affine"
    }
  end

  defp assert_equal(actual, expected) do
    assert Nx.shape(actual) == Nx.shape(expected)
    assert Nx.type(actual) == Nx.type(expected)
    assert Nx.to_number(Nx.all(Nx.equal(actual, expected))) == 1
  end

  defp assert_rel_close(actual, expected, tol) do
    actual = Nx.as_type(actual, :f32)
    err = actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    scale = expected |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert err / scale < tol, "relative error #{err / scale} exceeds #{tol}"
  end
end
