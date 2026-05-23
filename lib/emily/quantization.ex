defmodule Emily.Quantization do
  @moduledoc """
  Quantized inference primitives.

  ## Public API

    * `quantized_matmul/2` — eager-mode fused kernel over a
      materialized `%Emily.QuantizedWeight{}` and an `Nx.Tensor`.
      Calls the MLX `quantized_matmul` C++ kernel directly;
      produces `Nx.dot(x, to_dense(qw) |> Nx.transpose())` within
      quantization tolerance.
    * `dequantize_defn/1` — defn-native analogue of
      `Emily.QuantizedWeight.to_dense/1`, composed from
      `Nx.right_shift` / `Nx.bitwise_and` / multiply / add. Use inside
      `Nx.Defn.jit`-traced Axon forward passes where a fused
      `quantized_matmul` node isn't available; `Nx.dot(x,
      dequantize_defn(qw))` runs in two kernels (dequantize then
      dense matmul) instead of one.
    * `defn_supported_bits/0` — enumerates the bit widths the
      defn-native path supports (`#{inspect([2, 3, 4, 6, 8])}`).

  See `Emily.QuantizedWeight` for the container struct and
  `Emily.QuantizedWeight.from_dense/2` for building one.
  """

  import Nx.Defn

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Emily.QuantizedWeight
  alias Nx.Tensor, as: T

  @defn_supported_bits [2, 3, 4, 6, 8]

  @doc """
  Bit widths supported by `dequantize_defn/1` (and therefore by
  `Emily.Quantization.Layers.quantized_dense/4` and any Axon graph
  rewrite that wires it in).

  `bits ∈ {3, 6}` use cross-u32 lane packing and therefore take a
  denser unpacking path than the integral-lanes-per-u32 bit widths.

  ## Examples

      iex> Emily.Quantization.defn_supported_bits()
      [2, 3, 4, 6, 8]

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

      iex> w = Nx.iota({4, 128}, backend: Emily.Backend, type: :f32)
      iex> qw = Emily.QuantizedWeight.from_dense(w)
      iex> x = Nx.iota({3, 128}, backend: Emily.Backend, type: :f32)
      iex> y = Emily.Quantization.quantized_matmul(x, qw)
      iex> Nx.shape(y)
      {3, 4}

  """
  @spec quantized_matmul(Nx.Tensor.t(), QuantizedWeight.t()) :: Nx.Tensor.t()
  def quantized_matmul(%T{} = x, %QuantizedWeight{} = qw) do
    x = Nx.backend_transfer(x, Emily.Backend)
    validate_dtype_match!(x, qw)

    %T{data: %B{ref: x_ref}} = x

    %QuantizedWeight{
      value: %T{data: %B{ref: q_ref}},
      scales: %T{data: %B{ref: s_ref}},
      biases: biases,
      group_size: group_size,
      bits: bits,
      transpose: transpose,
      mode: mode
    } = qw

    b_ref = QuantizedWeight.biases_ref(mode, biases)

    w = Emily.MlxStream.default_worker()

    out_ref =
      Native.quantized_matmul(
        w,
        x_ref,
        q_ref,
        s_ref,
        b_ref,
        transpose,
        group_size,
        bits,
        mode
      )

    shape = out_ref |> Native.shape() |> List.to_tuple()
    type = Native.dtype(out_ref)

    %T{
      data: %B{ref: out_ref},
      shape: shape,
      type: type,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end

  # Affine: MLX's quantized_matmul requires x and scales to share a dtype.
  # Microscaled modes store scales as a u8 (e8m0 or e4m3 exponent);
  # MLX promotes internally, so just require `x` to be a real float.
  defp validate_dtype_match!(%T{type: x_type}, %QuantizedWeight{
         mode: "affine",
         scales: %T{type: s_type}
       })
       when x_type == s_type,
       do: :ok

  defp validate_dtype_match!(%T{type: x_type}, %QuantizedWeight{
         mode: "affine",
         scales: %T{type: s_type}
       }) do
    raise ArgumentError,
          "Emily.Quantization.quantized_matmul/2: input dtype #{inspect(x_type)} must " <>
            "match scales dtype #{inspect(s_type)}. Cast the input with " <>
            "`Nx.as_type/2` before calling."
  end

  defp validate_dtype_match!(%T{type: {:f, _}}, %QuantizedWeight{}), do: :ok
  defp validate_dtype_match!(%T{type: {:bf, _}}, %QuantizedWeight{}), do: :ok

  defp validate_dtype_match!(%T{type: x_type}, %QuantizedWeight{mode: mode}) do
    raise ArgumentError,
          "Emily.Quantization.quantized_matmul/2: microscaled mode #{inspect(mode)} " <>
            "requires a floating input dtype, got: #{inspect(x_type)}."
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
  pack across u32 boundaries and use a dense bitstream unpacking path.

  ## Examples

      iex> w = Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
      iex> qw = Emily.QuantizedWeight.from_dense(w, group_size: 64, bits: 4)
      iex> dense = Emily.Quantization.dequantize_defn(qw)
      iex> Nx.shape(dense)
      {4, 64}

  """
  @spec dequantize_defn(QuantizedWeight.t()) :: Nx.Tensor.t()
  deftransform dequantize_defn(qw) do
    %QuantizedWeight{
      value: q,
      scales: s,
      biases: b,
      group_size: group_size,
      bits: bits,
      mode: mode
    } = qw

    validate_defn_mode!(mode)
    validate_defn_bits!(bits)

    dequantize_impl(q, s, b, group_size: group_size, bits: bits)
  end

  defp validate_defn_mode!("affine"), do: :ok

  defp validate_defn_mode!(mode) do
    raise ArgumentError,
          "Emily.Quantization.dequantize_defn/1: mode=#{inspect(mode)} is not " <>
            "supported by the defn-native path (only \"affine\" is). " <>
            "Use `Emily.QuantizedWeight.to_dense/1` (the Native path) to " <>
            "dequantize microscaled modes."
  end

  defp validate_defn_bits!(bits) when bits in @defn_supported_bits, do: :ok

  defp validate_defn_bits!(bits) do
    raise ArgumentError,
          "Emily.Quantization.dequantize_defn/1: bits=#{bits} is not supported " <>
            "by the defn-native path. Supported: #{inspect(@defn_supported_bits)}. Use " <>
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

    masked =
      if rem(32, bits) == 0 do
        unpack_integral_lanes(w_q, bits)
      else
        unpack_cross_word_lanes(w_q, bits)
      end

    # Integral lanes unpack to (..., packed, lpu); cross-word lanes
    # already unpack to (..., orig_last). Regroup to (..., groups,
    # group_size) so per-group scale/bias broadcast trivially.
    grouped = masked |> flatten_unpacked(bits) |> group_last_axis(group_size)

    # Cast unpacked integers → scales dtype, then per-group affine
    # recombine; flatten back to (..., orig_last).
    grouped_f = Nx.as_type(grouped, Nx.type(scales))
    dequantized = grouped_f * Nx.new_axis(scales, -1) + Nx.new_axis(biases, -1)

    Nx.flatten(dequantized, axes: [-2, -1])
  end

  defnp unpack_integral_lanes(w_q, bits) do
    # Unpack: (..., packed) → (..., packed, lpu) via broadcast-shift
    # with a length-lpu shift vector, then mask to `bits`-width lanes.
    shifts = build_shifts(bits)
    mask = build_mask(bits)

    # new_axis appends a length-1 axis; right_shift broadcasts against shifts.
    w_q
    |> Nx.new_axis(-1)
    |> Nx.right_shift(shifts)
    |> Nx.bitwise_and(mask)
  end

  defnp unpack_cross_word_lanes(w_q, bits) do
    {word_indices, next_word_indices} = build_word_indices(w_q, bits)
    bit_indices = build_bit_indices(w_q, bits)

    w_q_u64 = Nx.as_type(w_q, :u64)
    current = Nx.take(w_q_u64, word_indices, axis: -1)
    next = Nx.take(w_q_u64, next_word_indices, axis: -1)

    next_shifted = Nx.left_shift(next, Nx.tensor(32, type: :u64))

    current
    |> Nx.bitwise_or(next_shifted)
    |> Nx.right_shift(bit_indices)
    |> Nx.bitwise_and(build_mask64(bits))
  end

  deftransformp flatten_unpacked(t, bits) do
    if rem(32, bits) == 0 do
      Nx.flatten(t, axes: [-2, -1])
    else
      t
    end
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

  deftransformp build_mask64(bits) do
    import Bitwise
    Nx.tensor((1 <<< bits) - 1, type: :u64)
  end

  deftransformp build_word_indices(w_q, bits) do
    shape = Nx.shape(w_q)
    rank = tuple_size(shape)
    packed = elem(shape, rank - 1)
    unpacked = div(packed * 32, bits)
    max_word = packed - 1

    words =
      for i <- 0..(unpacked - 1) do
        div(i * bits, 32)
      end

    next_words = Enum.map(words, &min(&1 + 1, max_word))

    {Nx.tensor(words, type: :s64), Nx.tensor(next_words, type: :s64)}
  end

  deftransformp build_bit_indices(w_q, bits) do
    shape = Nx.shape(w_q)
    rank = tuple_size(shape)
    packed = elem(shape, rank - 1)
    unpacked = div(packed * 32, bits)

    bit_indices =
      for i <- 0..(unpacked - 1) do
        rem(i * bits, 32)
      end

    Nx.tensor(bit_indices, type: :u64)
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
