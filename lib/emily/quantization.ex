defmodule Emily.Quantization do
  @moduledoc """
  Quantized inference primitives.

  M10 ships the Native bindings (`Emily.Native.quantize/3`,
  `Emily.Native.dequantize/5`, `Emily.Native.quantized_matmul/7`), the
  `Emily.QuantizedWeight` container, and this direct-call helper. This is
  enough to:

    * quantize a dense weight, store it packed at rest, and later dispatch a
      fused quantized matmul against it;
    * benchmark the quantized path against an `Nx.dot`-on-dequantized oracle;
    * run quantized inference in *eager* code (plain Elixir, outside `defn`).

  ## What M10 does NOT ship

  Integration with `Nx.Defn`-traced Axon forward passes (and therefore
  `Bumblebee` serving with AWQ checkpoints) is deferred to a follow-up.
  `Nx.Defn.Evaluator` walks `Nx.Defn.Expr` and dispatches via the
  `Nx.Backend` behaviour; there is no public hook to inject a custom op like
  `Native.quantized_matmul`, and none of `deftransform`, `hook`, or
  `metadata` can call NIFs on tensors that are still `Expr` nodes at trace
  time. Closing that gap requires either (a) a defn-native dequantize built
  from Nx bit primitives (skips the fused kernel), or (b) an Emily-specific
  compiler variant that recognizes a sentinel `Expr` node. Both are
  meaningful scope; M10.5 will tackle one of them.

  Use `quantized_matmul/2` below for any eager quantized inference today.
  """

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Emily.QuantizedWeight
  alias Nx.Tensor, as: T

  @doc """
  Compute `x @ W^T` where `W` is represented as a `QuantizedWeight`.

  With `qw.transpose == true` (the default from `QuantizedWeight.from_dense/2`)
  this matches `Nx.dot(x, QuantizedWeight.to_dense(qw) |> Nx.transpose())`
  — i.e. a dense-kernel dot with a pre-transposed, dequantized weight —
  within MLX's quantization tolerance. With `transpose == false`, MLX
  interprets the packed layout as already transposed (the AWQ convention).

  Both operands must live on `Emily.Backend`; pass scalars/tensors from
  `Nx.BinaryBackend` and they will be transferred. The input dtype must
  match `qw.scales.type` (typically f16, bf16, or f32).

  ## Examples

      w = Nx.iota({4, 128}, backend: Emily.Backend, type: :f32)
      qw = Emily.QuantizedWeight.from_dense(w)
      x = Nx.iota({3, 128}, backend: Emily.Backend, type: :f32)
      y = Emily.Quantization.quantized_matmul(x, qw)
      # y :: f32[3][4]
  """
  @spec quantized_matmul(Nx.Tensor.t(), QuantizedWeight.t()) :: Nx.Tensor.t()
  def quantized_matmul(%T{} = x, %QuantizedWeight{} = qw) do
    x = Nx.backend_transfer(x, Emily.Backend)
    validate_dtype_match!(x, qw)

    %T{data: %B{ref: x_ref}} = x

    %QuantizedWeight{
      value: %T{data: %B{ref: q_ref}},
      scales: %T{data: %B{ref: s_ref}},
      biases: %T{data: %B{ref: b_ref}},
      group_size: group_size,
      bits: bits,
      transpose: transpose
    } = qw

    out_ref =
      Native.quantized_matmul(x_ref, q_ref, s_ref, b_ref, transpose, group_size, bits)

    shape = out_ref |> Native.shape() |> List.to_tuple()
    type = Native.dtype(out_ref)

    %T{
      data: %B{ref: out_ref},
      shape: shape,
      type: type,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end

  # MLX's quantized_matmul kernel requires x and scales to share a dtype —
  # mismatches fail inside MLX with a less helpful message than this one.
  defp validate_dtype_match!(%T{type: x_type}, %QuantizedWeight{scales: %T{type: s_type}})
       when x_type == s_type,
       do: :ok

  defp validate_dtype_match!(%T{type: x_type}, %QuantizedWeight{scales: %T{type: s_type}}) do
    raise ArgumentError,
          "Emily.Quantization.quantized_matmul/2: input dtype #{inspect(x_type)} must " <>
            "match scales dtype #{inspect(s_type)}. Cast the input with " <>
            "`Nx.as_type/2` before calling."
  end
end
