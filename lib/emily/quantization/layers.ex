defmodule Emily.Quantization.Layers do
  @moduledoc """
  Defn-traceable quantized layer op for use inside Axon graphs.

  `quantized_dense/4` is the drop-in replacement for `Axon.Layers.dense/4`
  on a `%Emily.QuantizedWeight{}` kernel. See `Emily.Quantization` for
  the defn-integration trade-offs; the `qwen3_quantized` notebook walks
  through a concrete `Axon.rewrite_nodes/2`-based graph rewrite that
  swaps every `:dense` for a layer calling this op.
  """

  import Nx.Defn

  alias Emily.QuantizedWeight

  @doc """
  Axon layer op: `x @ W (+ bias)` where `W` is a `%QuantizedWeight{}`.

  Mirrors the signature of `Axon.Quantization.Layers.weight_only_quantized_dense/4`:

    * `input` — activation tensor, shape `(..., in)`.
    * `kernel` — `%QuantizedWeight{}`. The stored layout is determined
      by `kernel.transpose`:
        * `transpose: false` (the AWQ / Axon-native layout) — packed
          representation of a `[in, out]` weight; the layer computes
          `Nx.dot(x, dense)`.
        * `transpose: true` (the MLX / PyTorch-native layout, i.e. fresh
          output of `QuantizedWeight.from_dense/2` on a `[out, in]`
          weight) — packed representation of a `[out, in]` weight; the
          layer computes `Nx.dot(x, Nx.transpose(dense))`.
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
    {bias, opts} =
      case bias do
        b when is_list(b) -> {Nx.tensor(0), Keyword.merge(opts, b)}
        b -> {b, opts}
      end

    %QuantizedWeight{transpose: transpose} = kernel
    opts = Keyword.put(opts, :transpose, transpose)
    quantized_dense_impl(input, kernel, bias, opts)
  end

  # `transpose` is threaded through `opts` as a compile-time constant so
  # the branch selects at trace time (no runtime `if` over booleans).
  defnp quantized_dense_impl(x, kernel, bias, opts \\ []) do
    # `:mode` is injected by Axon's compiler (`:inference` / `:train`)
    # for every layer op; accept-and-ignore here since weight-only
    # quantization has no mode-dependent behavior.
    opts = keyword!(opts, [:transpose, mode: :inference])
    dense = Emily.Quantization.dequantize_defn(kernel)

    y =
      if opts[:transpose] do
        Nx.dot(x, Nx.transpose(dense))
      else
        Nx.dot(x, dense)
      end

    Nx.add(y, bias)
  end
end
