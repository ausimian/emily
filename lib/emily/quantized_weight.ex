defmodule Emily.QuantizedWeight do
  @moduledoc """
  Container for a matrix quantized via one of MLX's group-wise quantization
  schemes (`"affine"` int4/int8, plus the microscaled variants
  `"mxfp4"`, `"mxfp8"`, `"nvfp4"`).

  A `QuantizedWeight` bundles the packed integer weights with the per-group
  scale and bias tensors that are needed to recover (or multiply against)
  the original dense matrix. It derives `Nx.Container` so it can flow through
  defn transforms, parameter maps, and `backend_transfer` alongside regular
  tensors — while the `group_size`, `bits`, `transpose`, and `mode` metadata
  survive the traversal via `Nx.Container`'s `keep:` option.

  ## Layout

    * `:value` — packed integer weights. For `bits=4` the packing is 8 nibbles
      per `uint32`; the last axis therefore shrinks by a factor of `32 / bits`.
    * `:scales` — per-group scale. For `"affine"` its dtype matches the
      source weight's dtype and its shape is `(..., last_dim / group_size)`.
      Microscaled modes store a fused e8m0/e4m3 scale instead
      (dtype `:u8`).
    * `:biases` — per-group bias for `"affine"`. For microscaled modes
      MLX's `fp_quantize` doesn't emit biases; the field holds a
      scalar-zero placeholder so `Nx.Container` can still traverse it,
      and the Native layer substitutes `nil` before dispatching to MLX.
    * `:group_size` / `:bits` — the parameters originally passed to
      `MLX`'s `quantize`.
    * `:transpose` — `true` if the quantized matmul should treat `:value` as
      `[out, in]` (the MLX / fresh-`from_dense/2` default). External
      checkpoint formats (e.g. AWQ) may need `false`.
    * `:mode` — quantization mode string. `"affine"` (default) is the
      classical int4/int8 scheme with real `biases`. The microscaled
      modes (`"mxfp4"`, `"mxfp8"`, `"nvfp4"`) trade `biases` for a
      floating-point scale format and carry fixed `group_size` / `bits`
      combinations (see below).

  ## Microscaled modes

  MLX's microscaled variants (see `QuantizationMode` in `vendor/mlx/
  mlx/primitives.h`) each require a specific `group_size` and `bits`:

  | Mode     | group_size | bits |
  |----------|------------|------|
  | `mxfp4`  | 32         | 4    |
  | `mxfp8`  | 32         | 8    |
  | `nvfp4`  | 16         | 4    |

  Mismatches raise at `from_dense/2` time with a clear error before
  touching MLX. `dequantize_defn/1` only understands the affine format;
  microscaled modes must round-trip through `to_dense/1` (the Native
  path).

  ## Dispatch

  Use `Emily.Quantization.quantized_matmul/2` to run a fused quantized
  matmul against a `QuantizedWeight`. `Nx.dot/2` itself cannot accept
  a `QuantizedWeight` operand — `Nx` traverses containers expecting a
  single tensor — so the direct-call helper is the supported path.

  > #### Defn-traced Axon forward passes {: .info}
  > `Emily.Quantization.dequantize_defn/1` is the defn-native analogue
  > of `to_dense/1`; pair it with
  > `Emily.Quantization.Layers.quantized_dense/4` to splice a
  > quantized linear into any `Nx.Defn.jit`-traced Axon forward pass.
  > The layer performs `Nx.dot(x, dequantize_defn(qw))` instead of
  > the fused `quantized_matmul` kernel — two dispatches vs one, but
  > fully integrated with the rest of Bumblebee's defn graph. Only
  > `mode: "affine"` is supported on the defn path today.
  """

  @derive {Nx.Container,
           containers: [:value, :scales, :biases], keep: [:group_size, :bits, :transpose, :mode]}
  @enforce_keys [:value, :scales, :biases, :group_size, :bits, :transpose, :mode]
  defstruct [:value, :scales, :biases, :group_size, :bits, :transpose, :mode]

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Nx.Tensor, as: T

  @type mode :: String.t()

  @type t :: %__MODULE__{
          value: Nx.Tensor.t(),
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t(),
          group_size: pos_integer(),
          bits: pos_integer(),
          transpose: boolean(),
          mode: mode()
        }

  @valid_bits [2, 3, 4, 6, 8]
  @valid_input_types [{:f, 32}, {:f, 16}, {:bf, 16}]
  @valid_modes ~w(affine mxfp4 mxfp8 nvfp4)

  # Microscaled modes pin exact {group_size, bits}. See MLX's
  # `fp_quantize` in `vendor/mlx/mlx/ops.cpp:4801-4823` for the canonical
  # constraint checks; we mirror them here so misuse fails with an Emily
  # error pointing at the option, rather than a deep C++ `invalid_argument`.
  @microscaled_constraints %{
    "mxfp4" => {32, 4},
    "mxfp8" => {32, 8},
    "nvfp4" => {16, 4}
  }

  @doc """
  Quantize a dense float tensor into a `QuantizedWeight`.

  The input must live on `Emily.Backend` (transfer first if you're coming
  from `Nx.BinaryBackend`). The last axis must be divisible by `:group_size`.

  ## Options

    * `:group_size` — default `64`. Elements per quantization group.
      Microscaled modes pin this to a specific value (see
      `Emily.QuantizedWeight` moduledoc).
    * `:bits` — default `4`. One of `#{inspect(@valid_bits)}` for
      `"affine"`; pinned by the mode for microscaled variants.
    * `:transpose` — default `true`. Layout flag threaded to the
      `quantized_matmul` kernel. Leave as `true` for weights produced
      here; set `false` when wrapping pre-packed external checkpoints.
    * `:mode` — default `"affine"`. One of `#{inspect(@valid_modes)}`.

  ## Examples

      iex> w = Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
      iex> qw = Emily.QuantizedWeight.from_dense(w, bits: 4, group_size: 64)
      iex> qw.bits
      4
      iex> qw.group_size
      64
      iex> qw.mode
      "affine"
      iex> Nx.shape(qw.value)
      {4, 8}

  """
  @spec from_dense(Nx.Tensor.t(), keyword()) :: t()
  def from_dense(%T{} = w, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        group_size: 64,
        bits: 4,
        transpose: true,
        mode: "affine"
      )

    group_size = opts[:group_size]
    bits = opts[:bits]
    transpose = opts[:transpose]
    mode = opts[:mode]

    validate_mode!(mode)
    validate_bits!(bits)
    validate_microscaled_constraints!(mode, group_size, bits)
    validate_input_type!(w)
    validate_last_axis!(w, group_size)

    w = Nx.backend_transfer(w, Emily.Backend)
    %T{data: %B{ref: w_ref}} = w

    {q_ref, s_ref, b_ref} =
      Native.quantize(Emily.MlxStream.default_worker(), w_ref, group_size, bits, mode)

    %__MODULE__{
      value: ref_to_tensor(q_ref),
      scales: ref_to_tensor(s_ref),
      biases: ref_to_tensor(b_ref),
      group_size: group_size,
      bits: bits,
      transpose: transpose,
      mode: mode
    }
  end

  @doc """
  Reconstruct a dense `Nx.Tensor` from a `QuantizedWeight`.

  Useful for oracle comparisons and for transferring a quantized
  parameter off `Emily.Backend` (backend transfer traverses each
  container tensor individually; most consumers want the dense view).

  ## Examples

      iex> w = Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
      iex> dense = Emily.QuantizedWeight.from_dense(w) |> Emily.QuantizedWeight.to_dense()
      iex> Nx.shape(dense)
      {4, 64}

  """
  @spec to_dense(t()) :: Nx.Tensor.t()
  def to_dense(%__MODULE__{
        value: %T{data: %B{ref: q}},
        scales: %T{data: %B{ref: s}},
        biases: biases,
        group_size: group_size,
        bits: bits,
        mode: mode
      }) do
    b_ref = biases_ref(mode, biases)

    ref =
      Native.dequantize(
        Emily.MlxStream.default_worker(),
        q,
        s,
        b_ref,
        group_size,
        bits,
        mode
      )

    ref_to_tensor(ref)
  end

  @doc false
  # Returns the NIF-ready biases reference for the given mode:
  # a raw ref for "affine", `nil` for microscaled modes (where the
  # container holds a placeholder tensor that MLX doesn't accept).
  @spec biases_ref(String.t(), Nx.Tensor.t() | nil) :: reference() | nil
  def biases_ref("affine", %T{data: %B{ref: ref}}), do: ref
  def biases_ref(_mode, _biases), do: nil

  # -- helpers ------------------------------------------------------

  defp validate_mode!(mode) when mode in @valid_modes, do: :ok

  defp validate_mode!(mode) do
    raise ArgumentError,
          "Emily.QuantizedWeight: :mode must be one of #{inspect(@valid_modes)}, " <>
            "got: #{inspect(mode)}"
  end

  defp validate_bits!(bits) when bits in @valid_bits, do: :ok

  defp validate_bits!(bits) do
    raise ArgumentError,
          "Emily.QuantizedWeight: :bits must be one of #{inspect(@valid_bits)}, got: #{inspect(bits)}"
  end

  defp validate_microscaled_constraints!("affine", _group_size, _bits), do: :ok

  defp validate_microscaled_constraints!(mode, group_size, bits) do
    {expected_gs, expected_bits} = Map.fetch!(@microscaled_constraints, mode)

    cond do
      group_size != expected_gs ->
        raise ArgumentError,
              "Emily.QuantizedWeight: mode #{inspect(mode)} requires group_size=#{expected_gs}, " <>
                "got: #{inspect(group_size)}"

      bits != expected_bits ->
        raise ArgumentError,
              "Emily.QuantizedWeight: mode #{inspect(mode)} requires bits=#{expected_bits}, " <>
                "got: #{inspect(bits)}"

      true ->
        :ok
    end
  end

  defp validate_input_type!(%T{type: type}) when type in @valid_input_types, do: :ok

  defp validate_input_type!(%T{type: type}) do
    raise ArgumentError,
          "Emily.QuantizedWeight.from_dense/2 requires #{inspect(@valid_input_types)}, " <>
            "got: #{inspect(type)}"
  end

  defp validate_last_axis!(%T{shape: shape}, group_size) when tuple_size(shape) >= 2 do
    last = elem(shape, tuple_size(shape) - 1)

    if rem(last, group_size) != 0 do
      raise ArgumentError,
            "Emily.QuantizedWeight: last axis (#{last}) must be divisible by " <>
              ":group_size (#{group_size}); shape=#{inspect(shape)}"
    end

    :ok
  end

  defp validate_last_axis!(%T{shape: shape}, _group_size) do
    raise ArgumentError,
          "Emily.QuantizedWeight.from_dense/2 requires rank ≥ 2 input (MLX " <>
            "quantization constraint), got shape #{inspect(shape)}"
  end

  defp ref_to_tensor(ref) do
    shape = ref |> Native.shape() |> List.to_tuple()
    type = Native.dtype(ref)

    %T{
      data: %B{ref: ref},
      shape: shape,
      type: type,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end
end
