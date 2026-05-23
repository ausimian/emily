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
  The math is identical to MLX's `dequantize`: lane `i` is extracted
  from the packed u32 stream at bit offset `i * bits`, masked to the
  low `bits` bits, then `w[i] = lane * scales[g] + biases[g]` where
  `g = div(i, group_size)` is the group index along the last axis.

  Supported: `bits ∈ #{inspect(@defn_supported_bits)}`. Two unpack
  paths are picked at trace time:

    * `bits ∈ {2, 4, 8}` — integral lanes per u32, broadcast-shift
      `w[..., :, lane] = (w_q[..., :] >> (lane * bits)) & mask`.
    * `bits ∈ {3, 6}` — lanes cross u32 boundaries, so we read
      adjacent u32 pairs as a u64, then shift by `rem(i * bits, 32)`
      and mask.

  Supported modes: `"affine"` and `"mxfp4"`. `mxfp4` decodes each
  4-bit lane through MLX's FP4-E2M1 lookup table and each u8 scale
  byte through `2^(s - 127)` (FP8-E8M0); the output dtype is `bf16`
  to match `QuantizedWeight.to_dense/1`. `mxfp8` and `nvfp4` are not
  yet wired and still raise — use the Native path for those.

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

    case mode do
      "affine" -> dequantize_impl(q, s, b, group_size: group_size, bits: bits)
      "mxfp4" -> dequantize_mxfp4_impl(q, s, group_size: group_size)
    end
  end

  defp validate_defn_mode!(mode) when mode in ["affine", "mxfp4"], do: :ok

  @supported_defn_modes ~w[affine mxfp4]

  defp validate_defn_mode!(mode) do
    raise ArgumentError,
          "Emily.Quantization.dequantize_defn/1: mode=#{inspect(mode)} is not " <>
            "supported by the defn-native path (supported: " <>
            "#{inspect(@supported_defn_modes)}). Use " <>
            "`Emily.QuantizedWeight.to_dense/1` (the Native path) to " <>
            "dequantize the remaining microscaled modes."
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

    # `bits` must be a trace-time integer (it comes from the
    # `%QuantizedWeight{}` `:keep` container metadata, not a tensor),
    # so the `if` constant-folds to a single branch. The two arms
    # return tensors of DIFFERENT ranks (integral adds a `new_axis`;
    # cross-word preserves rank) — if a future refactor promotes
    # `bits` to a runtime tensor, defn's cond shape-compatibility
    # check will reject the mismatch.
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

  # MLX `mxfp4`: bits=4 lanes packed in u32 stream, scales=u8 FP8-E8M0
  # exponent bytes (one per group of 32 lanes), no biases. Lane codes
  # decode through a 16-entry FP4-E2M1 LUT; scale bytes decode through
  # `2^(s - 127)` (a 256-entry LUT built once at trace time).
  #
  # Output dtype is bf16 to match `QuantizedWeight.to_dense/1` on mxfp4.
  # All values involved (FP4 lane LUT entries, E8M0 scale powers, and
  # their products) are exact in bf16 for realistic inputs, so the
  # defn path is bit-identical to MLX's NIF dequant in practice.
  defnp dequantize_mxfp4_impl(w_q, scales, opts \\ []) do
    opts = keyword!(opts, [:group_size])
    group_size = opts[:group_size]

    # Unpack 4-bit lane codes: bits=4 → lpu=8, rem(32, 4) == 0 so the
    # integral path applies. Result shape: (..., packed, 8).
    lane_codes =
      w_q
      |> unpack_integral_lanes(4)
      |> Nx.flatten(axes: [-2, -1])

    # FP4-E2M1 decode via 16-entry LUT.
    lanes_f = Nx.take(fp4_lut(), lane_codes)

    # Group: (..., orig_last) → (..., groups, group_size) so scales
    # broadcast cleanly via a trailing length-1 axis.
    grouped = group_last_axis(lanes_f, group_size)

    # FP8-E8M0 scale decode via 256-entry LUT; result shape matches
    # scales (..., groups), then add the length-1 axis for broadcast.
    scales_f = Nx.take(e8m0_lut(), scales)

    dequantized = grouped * Nx.new_axis(scales_f, -1)
    Nx.flatten(dequantized, axes: [-2, -1])
  end

  deftransformp fp4_lut do
    # MLX FP4-E2M1 lane table (sign bit at code bit 3, low 3 bits index
    # the magnitude). Matches `FP4_LUT` in `deps/mlx_src/mlx/backend/cpu/quantized.cpp`.
    Nx.tensor(
      [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0
      ],
      type: {:bf, 16}
    )
  end

  deftransformp e8m0_lut do
    # FP8-E8M0: 8 exponent bits, no sign or mantissa; value = 2^(s - 127).
    # `s = 0xFF` is NaN per the OCP MX spec; `s = 0x00` is the subnormal
    # representation of zero. The formula below produces a very large
    # finite value at s=255 (saturates to +inf in bf16) and 2^-127 at
    # s=0 (subnormal in bf16, but realistic MLX scales never go that
    # low). Practical correctness holds for every scale MLX emits.
    values = for s <- 0..255, do: :math.pow(2.0, s - 127)
    Nx.tensor(values, type: {:bf, 16})
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

    # Invariant: `unpacked * bits == packed * 32` (no leftover bits at
    # the end of the buffer). Equivalent to "the final lane ends
    # exactly at the end of the final u32, so no lane in the final
    # word straddles past it." This holds for every (bits, group_size)
    # combo MLX accepts; `QuantizedWeight.from_dense/2` rejects shapes
    # that violate it. The `min(&1 + 1, max_word)` clamp below is
    # correctness-safe ONLY under this invariant — for lanes whose
    # `next_word` is clamped to themselves, the duplicated high bits
    # never participate because the lane's bit range fits in the
    # current word and is masked off after the shift.
    if rem(packed * 32, bits) != 0 do
      raise ArgumentError,
            "Emily.Quantization.dequantize_defn/1: packed length #{packed} " <>
              "is incompatible with bits=#{bits} (rem(packed*32, bits) != 0). " <>
              "MLX packing requires orig_last * bits to be a multiple of 32."
    end

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
