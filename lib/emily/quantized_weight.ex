defmodule Emily.QuantizedWeight do
  @moduledoc """
  Container for a matrix quantized via MLX affine group-wise quantization.

  A `QuantizedWeight` bundles the packed integer weights with the per-group
  scale and bias tensors that are needed to recover (or multiply against)
  the original dense matrix. It derives `Nx.Container` so it can flow through
  defn transforms, parameter maps, and `backend_transfer` alongside regular
  tensors — while the `group_size`, `bits`, and `transpose` metadata survive
  the traversal via `Nx.Container`'s `keep:` option.

  ## Layout

    * `:value` — packed integer weights. For `bits=4` the packing is 8 nibbles
      per `uint32`; the last axis therefore shrinks by a factor of `32 / bits`.
    * `:scales` — per-group scale, dtype matches the source weight's dtype,
      shape `(..., last_dim / group_size)`.
    * `:biases` — per-group bias, same dtype/shape as `:scales`.
    * `:group_size` / `:bits` — the parameters originally passed to
      `MLX`'s `quantize`.
    * `:transpose` — `true` if the quantized matmul should treat `:value` as
      `[out, in]` (the MLX / fresh-`from_dense/2` default). External
      checkpoint formats (e.g. AWQ) may need `false`.

  ## Dispatch

  Use `Emily.Quantization.quantized_matmul/2` to run a fused quantized matmul
  against a `QuantizedWeight`. `Nx.dot/2` itself cannot accept a
  `QuantizedWeight` operand — `Nx` traverses containers expecting a single
  tensor — so the direct-call helper is the supported path.

  > #### Defn-traced Axon forward passes {: .info}
  > `Emily.Quantization.dequantize_defn/1` (M10.5) is the defn-native
  > analogue of `to_dense/1`; pair it with
  > `Emily.Quantization.Layers.quantized_dense/4` to splice a quantized
  > linear into any `Nx.Defn.jit`-traced Axon forward pass. The layer
  > performs `Nx.dot(x, dequantize_defn(qw))` instead of MLX's single
  > fused `quantized_matmul` — two kernels vs. one, but fully
  > integrated with the rest of Bumblebee's defn graph. M11's fast-kernel
  > work will close the perf gap.
  """

  @derive {Nx.Container,
           containers: [:value, :scales, :biases], keep: [:group_size, :bits, :transpose]}
  @enforce_keys [:value, :scales, :biases, :group_size, :bits, :transpose]
  defstruct [:value, :scales, :biases, :group_size, :bits, :transpose]

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Nx.Tensor, as: T

  @type t :: %__MODULE__{
          value: Nx.Tensor.t(),
          scales: Nx.Tensor.t(),
          biases: Nx.Tensor.t(),
          group_size: pos_integer(),
          bits: pos_integer(),
          transpose: boolean()
        }

  @valid_bits [2, 3, 4, 6, 8]
  @valid_input_types [{:f, 32}, {:f, 16}, {:bf, 16}]

  @doc """
  Quantize a dense float tensor into a `QuantizedWeight`.

  The input must live on `Emily.Backend` (transfer first if you're coming
  from `Nx.BinaryBackend`). The last axis must be divisible by `:group_size`.

  ## Options

    * `:group_size` — default `64`. Elements per quantization group.
    * `:bits` — default `4`. One of `#{inspect(@valid_bits)}`.
    * `:transpose` — default `true`. Layout flag threaded to
      `Native.quantized_matmul/7`. Leave as `true` for weights produced
      here; set `false` when wrapping pre-packed external checkpoints.
  """
  @spec from_dense(Nx.Tensor.t(), keyword()) :: t()
  def from_dense(%T{} = w, opts \\ []) do
    opts = Keyword.validate!(opts, group_size: 64, bits: 4, transpose: true)
    group_size = opts[:group_size]
    bits = opts[:bits]
    transpose = opts[:transpose]

    validate_bits!(bits)
    validate_input_type!(w)
    validate_last_axis!(w, group_size)

    w = Nx.backend_transfer(w, Emily.Backend)
    %T{data: %B{ref: w_ref}} = w

    {q_ref, s_ref, b_ref} =
      Native.quantize(Emily.MlxStream.default_worker(), w_ref, group_size, bits)

    %__MODULE__{
      value: ref_to_tensor(q_ref),
      scales: ref_to_tensor(s_ref),
      biases: ref_to_tensor(b_ref),
      group_size: group_size,
      bits: bits,
      transpose: transpose
    }
  end

  @doc """
  Reconstruct a dense `Nx.Tensor` from a `QuantizedWeight`.

  Useful for oracle comparisons and for transferring a quantized parameter
  off `Emily.Backend` (backend transfer traverses each container tensor
  individually; most consumers want the dense view).
  """
  @spec to_dense(t()) :: Nx.Tensor.t()
  def to_dense(%__MODULE__{
        value: %T{data: %B{ref: q}},
        scales: %T{data: %B{ref: s}},
        biases: %T{data: %B{ref: b}},
        group_size: group_size,
        bits: bits
      }) do
    ref = Native.dequantize(Emily.MlxStream.default_worker(), q, s, b, group_size, bits)
    ref_to_tensor(ref)
  end

  # -- helpers ------------------------------------------------------

  defp validate_bits!(bits) when bits in @valid_bits, do: :ok

  defp validate_bits!(bits) do
    raise ArgumentError,
          "Emily.QuantizedWeight: :bits must be one of #{inspect(@valid_bits)}, got: #{inspect(bits)}"
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
