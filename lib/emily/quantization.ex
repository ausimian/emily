defmodule Emily.Quantization do
  @moduledoc """
  Quantized inference primitives.

  Two entry points:

    * `quantized_matmul/2` — eager-mode fused kernel (materialized
      tensors only). Extracts refs from a `%QuantizedWeight{}` and
      calls `Native.quantized_matmul/7`.
    * `dequantize_defn/1` — defn-native analogue of
      `QuantizedWeight.to_dense/1`, composed from `Nx.right_shift` /
      `Nx.bitwise_and` / multiply / add. Use inside `Nx.Defn.jit`-traced
      Axon forward passes; `Nx.dot(x, dequantize_defn(qw))` replaces
      the fused `quantized_matmul` kernel with two dispatches — M11's
      fast-kernel work closes that gap.

  `dequantize_defn/1` supports `bits ∈ #{inspect([2, 4, 8])}`;
  `bits ∈ {3, 6}` use cross-u32 lane packing (out of scope here — use
  `QuantizedWeight.to_dense/1`).

  See `Emily.Quantization.Layers.quantized_dense/4` for the
  Axon-compatible layer op built on `dequantize_defn/1`.
  """

  import Nx.Defn

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Emily.QuantizedWeight
  alias Nx.Tensor, as: T

  @defn_supported_bits [2, 4, 8]

  @doc """
  Bit widths supported by `dequantize_defn/1` (and therefore by
  `Emily.Quantization.Layers.quantized_dense/4` /
  `Emily.Quantization.Transform`).

  `bits ∈ {3, 6}` use cross-u32 lane packing and aren't supported by
  the defn-native path; `QuantizedWeight.to_dense/1` (the Native path)
  still handles them.
  """
  @spec defn_supported_bits() :: [pos_integer()]
  def defn_supported_bits, do: @defn_supported_bits

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

    s = Process.get(:emily_stream, -1)

    out_ref =
      Native.quantized_matmul(x_ref, q_ref, s_ref, b_ref, transpose, group_size, bits, s)

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

  # ================================================================
  # Defn-native dequantize
  # ================================================================

  @doc """
  Reconstruct a dense tensor from a `QuantizedWeight`, built entirely
  from Nx primitives so it composes inside `defn` traces.

  This is the defn-compatible analogue of `QuantizedWeight.to_dense/1`.
  The math is identical to MLX's `dequantize`:

      w[i] = (w_q_packed >> ((i mod lpu) * bits)) & mask * scales[g] + biases[g]

  where `lpu = div(32, bits)` (lanes per u32), `mask = (1 <<< bits) - 1`,
  and `g = div(i, group_size)` is the group index along the last axis.

  Supported: `bits ∈ #{inspect(@defn_supported_bits)}`. `bits ∈ {3, 6}`
  pack across u32 boundaries and are out of scope here.

  ## Example

      qw = Emily.QuantizedWeight.from_dense(w, group_size: 64, bits: 4)
      dense_defn = Emily.Quantization.dequantize_defn(qw)
      dense_native = Emily.QuantizedWeight.to_dense(qw)
      # element-wise identical
  """
  @spec dequantize_defn(QuantizedWeight.t()) :: Nx.Tensor.t()
  deftransform dequantize_defn(qw) do
    %QuantizedWeight{
      value: q,
      scales: s,
      biases: b,
      group_size: group_size,
      bits: bits
    } = qw

    validate_defn_bits!(bits)

    dequantize_impl(q, s, b, group_size: group_size, bits: bits)
  end

  defp validate_defn_bits!(bits) when bits in @defn_supported_bits, do: :ok

  defp validate_defn_bits!(bits) do
    raise ArgumentError,
          "Emily.Quantization.dequantize_defn/1: bits=#{bits} uses cross-u32 " <>
            "lane packing, which is out of scope for the defn-native path. " <>
            "Supported: #{inspect(@defn_supported_bits)}. Use " <>
            "`Emily.QuantizedWeight.to_dense/1` (the Native path) for unsupported " <>
            "bit widths."
  end

  # Expects `opts` to carry compile-time `:group_size` and `:bits`. Both
  # are used for shape arithmetic (lanes-per-u32, per-group reshape) so
  # they must be trace-time constants.
  defnp dequantize_impl(w_q, scales, biases, opts \\ []) do
    opts = keyword!(opts, [:group_size, :bits])
    group_size = opts[:group_size]
    bits = opts[:bits]

    # Unpack: (..., packed) → (..., packed, lpu) via broadcast-shift
    # with a length-lpu shift vector, then mask to `bits`-width nibbles.
    shifts = build_shifts(bits)
    mask = build_mask(bits)

    # new_axis appends a length-1 axis; right_shift broadcasts against shifts.
    shifted = Nx.right_shift(Nx.new_axis(w_q, -1), shifts)
    masked = Nx.bitwise_and(shifted, mask)

    # Flatten (packed, lpu) → orig_last, then regroup to (..., groups,
    # group_size) so per-group scale/bias broadcast trivially.
    grouped = masked |> Nx.flatten(axes: [-2, -1]) |> group_last_axis(group_size)

    # Cast u32 → scales dtype, then per-group affine recombine; flatten
    # back to (..., orig_last).
    grouped_f = Nx.as_type(grouped, Nx.type(scales))
    dequantized = grouped_f * Nx.new_axis(scales, -1) + Nx.new_axis(biases, -1)

    Nx.flatten(dequantized, axes: [-2, -1])
  end

  deftransformp build_shifts(bits) do
    lpu = div(32, bits)
    shifts = for i <- 0..(lpu - 1), do: i * bits
    Nx.tensor(shifts, type: :u32)
  end

  deftransformp build_mask(bits) do
    import Bitwise
    Nx.tensor((1 <<< bits) - 1, type: :u32)
  end

  # Reshape `(..., n)` → `(..., n / group_size, group_size)`. Uses
  # `:auto` so we don't recompute the quotient.
  deftransformp group_last_axis(t, group_size) do
    shape = Nx.shape(t)
    rank = tuple_size(shape)
    new_shape = shape |> put_elem(rank - 1, :auto) |> Tuple.insert_at(rank, group_size)
    Nx.reshape(t, new_shape)
  end
end
