defmodule Emily.Quantization.Layers do
  @moduledoc """
  Defn-traceable quantized layer op for use inside Axon graphs.

  `quantized_dense/4` is the drop-in replacement for `Axon.Layers.dense/4`
  on a `%Emily.QuantizedWeight{}` kernel. It lowers to the fused
  `mx::quantized_matmul` kernel (via `Emily.Quantization.quantized_matmul_defn/2`),
  which streams the packed low-bit weights directly rather than
  materializing a dense weight per call — the single-kernel path decode
  loops want. The `qwen3_quantized` notebook walks through a concrete
  `Axon.rewrite_nodes/2`-based graph rewrite that swaps every `:dense`
  for a layer calling this op.
  """

  import Nx.Defn

  alias Emily.QuantizedWeight

  @doc """
  Axon layer op: `x @ W (+ bias)` where `W` is a `%QuantizedWeight{}`.

  Mirrors the signature of `Axon.Quantization.Layers.weight_only_quantized_dense/4`:

    * `input` — activation tensor, shape `(..., in)`.
    * `kernel` — `%QuantizedWeight{}`. The stored layout is determined
      by `kernel.transpose` (passed straight through to the fused
      kernel):
        * `transpose: false` (the AWQ / Axon-native layout) — packed
          representation of a `[in, out]` weight; the layer computes
          `x @ W`.
        * `transpose: true` (the MLX / PyTorch-native layout, i.e. fresh
          output of `QuantizedWeight.from_dense/2` on a `[out, in]`
          weight) — packed representation of a `[out, in]` weight; the
          layer computes `x @ Wᵀ`.
    * `bias` — either an `Nx.Tensor`, a number, or a keyword list (in
      which case it's treated as `opts` and bias defaults to 0). Matches
      `Axon.Quantization.Layers.weight_only_quantized_dense/4`'s
      signature for drop-in use under `Axon.layer/3`.
    * `opts` — reserved for Axon-layer metadata; not used by this
      implementation directly (all state lives on the
      `%QuantizedWeight{}`).

  ## Examples

      iex> w = Nx.iota({4, 128}, backend: Emily.Backend, type: :f32)
      iex> qw = Emily.QuantizedWeight.from_dense(w)
      iex> x = Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
      iex> y = Emily.Quantization.Layers.quantized_dense(x, qw)
      iex> Nx.shape(y)
      {2, 4}

  """
  deftransform quantized_dense(input, kernel, bias \\ 0, opts \\ []) do
    # When Axon.dense registers `use_bias: false`, the generated op call
    # is arity-3 with layer opts as the third arg (matches
    # `Axon.Quantization.Layers.weight_only_quantized_dense/4`'s contract).
    # Axon also injects `:mode` (`:inference` / `:train`); weight-only
    # quantization has no mode-dependent behaviour, so `opts` is absorbed
    # and ignored here.
    {bias, _opts} =
      case bias do
        b when is_list(b) -> {Nx.tensor(0), Keyword.merge(opts, b)}
        b -> {b, opts}
      end

    # Assert the kernel is a %QuantizedWeight{} at the layer boundary so a
    # bad kernel fails here rather than deep inside the fused kernel. Its
    # layout/mode/bits/group_size are read off the struct by
    # `quantized_matmul_defn/2`, so nothing extra needs threading through.
    %QuantizedWeight{} = kernel
    quantized_dense_impl(input, kernel, bias)
  end

  defnp quantized_dense_impl(x, kernel, bias) do
    # Fused single-kernel `mx::quantized_matmul` — streams the 4-bit
    # packed weights directly — instead of dequantizing the full weight
    # to bf16 and then running a dense `Nx.dot`. Decode is
    # memory-bandwidth bound on the weight, so this is ~2-4x faster per
    # matmul (larger gains on the fatter MLP projections). `transpose`,
    # `mode`, `bits`, and `group_size` are read off the `%QuantizedWeight{}`
    # by `quantized_matmul_defn/2`; non-Emily backends still get the
    # composed `dequantize_defn/1` + `Nx.dot/2` via the block's fallback.
    Emily.Quantization.quantized_matmul_defn(x, kernel)
    |> Nx.add(bias)
  end
end
