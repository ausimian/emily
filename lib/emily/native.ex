defmodule Emily.Native do
  @moduledoc false
  # Thin NIF loader for the emily C++ shim. Every function here maps
  # directly to one NIF in c_src/. No policy, no caching, no defaults —
  # higher layers do that.
  #
  # Op NIFs take a leading `worker` parameter (reference to a
  # WorkerThread resource). The worker dispatches work to a dedicated OS
  # thread that owns the MLX stream. Core NIFs (from_binary, shape,
  # dtype) and memory introspection NIFs don't take a worker.
  #
  # The async model (see docs/planning/async-worker-exploration.md and
  # c_src/emily/async.hpp) has each worker-bound op NIF enqueue its
  # work onto the worker thread and return a ref immediately; the
  # worker posts `{ref, {:ok, result}}` or `{ref, {:error, reason}}`
  # back to the caller via `enif_send`. The public wrappers below
  # (`def foo(...)`) hide the ref + receive behind a synchronous
  # `Async.call/1` so callers see the same blocking semantics as the
  # prior sync path.

  alias Emily.Native.Async

  @on_load :__on_load__
  @compile {:autoload, false}

  @doc false
  def __on_load__ do
    path = :filename.join(:code.priv_dir(:emily), ~c"libemily")
    :erlang.load_nif(path, 0)
  end

  @type tensor :: reference()
  @type worker :: reference()
  @type dtype :: Nx.Type.t()

  defp nif, do: :erlang.nif_error(:nif_not_loaded)

  # --- Core --------------------------------------------------------

  @spec from_binary(binary(), [non_neg_integer()], dtype()) :: tensor()
  def from_binary(_data, _shape, _dtype), do: nif()

  @doc false
  @spec to_binary_nif(worker(), tensor()) :: reference()
  def to_binary_nif(_w, _tensor), do: nif()

  @spec to_binary(worker(), tensor()) :: binary()
  def to_binary(w, tensor), do: Async.call(to_binary_nif(w, tensor))

  @spec shape(tensor()) :: [non_neg_integer()]
  def shape(_tensor), do: nif()

  @spec dtype(tensor()) :: dtype()
  def dtype(_tensor), do: nif()

  @doc false
  @spec eval_nif(worker(), tensor()) :: reference()
  def eval_nif(_w, _tensor), do: nif()

  @spec eval(worker(), tensor()) :: :ok
  def eval(w, tensor), do: Async.call(eval_nif(w, tensor))

  # --- Worker ------------------------------------------------------

  @spec create_worker() :: worker()
  def create_worker, do: nif()

  # --- Creation ----------------------------------------------------

  @doc false
  @spec zeros_nif(worker(), [non_neg_integer()], dtype()) :: reference()
  def zeros_nif(_w, _shape, _dtype), do: nif()

  @spec zeros(worker(), [non_neg_integer()], dtype()) :: tensor()
  def zeros(w, shape, dtype), do: Async.call(zeros_nif(w, shape, dtype))

  @doc false
  @spec ones_nif(worker(), [non_neg_integer()], dtype()) :: reference()
  def ones_nif(_w, _shape, _dtype), do: nif()

  @spec ones(worker(), [non_neg_integer()], dtype()) :: tensor()
  def ones(w, shape, dtype), do: Async.call(ones_nif(w, shape, dtype))

  @doc false
  @spec full_nif(worker(), [non_neg_integer()], tensor(), dtype()) :: reference()
  def full_nif(_w, _shape, _value, _dtype), do: nif()

  @spec full(worker(), [non_neg_integer()], tensor(), dtype()) :: tensor()
  def full(w, shape, value, dtype), do: Async.call(full_nif(w, shape, value, dtype))

  @doc false
  @spec arange_nif(worker(), float(), float(), float(), dtype()) :: reference()
  def arange_nif(_w, _start, _stop, _step, _dtype), do: nif()

  @spec arange(worker(), float(), float(), float(), dtype()) :: tensor()
  def arange(w, start, stop, step, dtype),
    do: Async.call(arange_nif(w, start, stop, step, dtype))

  @doc false
  @spec eye_nif(worker(), integer(), integer(), integer(), dtype()) :: reference()
  def eye_nif(_w, _n, _m, _k, _dtype), do: nif()

  @spec eye(worker(), integer(), integer(), integer(), dtype()) :: tensor()
  def eye(w, n, m, k, dtype), do: Async.call(eye_nif(w, n, m, k, dtype))

  # --- Cast --------------------------------------------------------

  @doc false
  @spec astype_nif(worker(), tensor(), dtype()) :: reference()
  def astype_nif(_w, _a, _dtype), do: nif()

  @spec astype(worker(), tensor(), dtype()) :: tensor()
  def astype(w, a, dtype), do: Async.call(astype_nif(w, a, dtype))

  @doc false
  @spec bitcast_nif(worker(), tensor(), dtype()) :: reference()
  def bitcast_nif(_w, _a, _dtype), do: nif()

  @spec bitcast(worker(), tensor(), dtype()) :: tensor()
  def bitcast(w, a, dtype), do: Async.call(bitcast_nif(w, a, dtype))

  # --- Unary -------------------------------------------------------

  unary_ops = [
    :negative,
    :abs,
    :sign,
    :floor,
    :ceil,
    :sqrt,
    :rsqrt,
    :exp,
    :expm1,
    :log,
    :log1p,
    :log2,
    :log10,
    :sin,
    :cos,
    :tan,
    :arcsin,
    :arccos,
    :arctan,
    :sinh,
    :cosh,
    :tanh,
    :arcsinh,
    :arccosh,
    :arctanh,
    :sigmoid,
    :erf,
    :erfinv,
    :square,
    :reciprocal,
    :logical_not,
    :bitwise_invert,
    :isnan,
    :isinf,
    :isfinite,
    :conjugate,
    :real,
    :imag,
    :stop_gradient
  ]

  for op <- unary_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(worker(), tensor()) :: reference()
    def unquote(nif_name)(_w, _a), do: nif()

    @doc false
    @spec unquote(op)(worker(), tensor()) :: tensor()
    def unquote(op)(w, a), do: Async.call(unquote(nif_name)(w, a))
  end

  @doc false
  @spec round_nif(worker(), tensor(), integer()) :: reference()
  def round_nif(_w, _a, _decimals), do: nif()

  @spec round(worker(), tensor(), integer()) :: tensor()
  def round(w, a, decimals), do: Async.call(round_nif(w, a, decimals))

  # --- Binary ------------------------------------------------------

  binary_ops = [
    :add,
    :subtract,
    :multiply,
    :divide,
    :floor_divide,
    :remainder,
    :power,
    :maximum,
    :minimum,
    :logaddexp,
    :arctan2,
    :equal,
    :not_equal,
    :less,
    :less_equal,
    :greater,
    :greater_equal,
    :logical_and,
    :logical_or,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :left_shift,
    :right_shift
  ]

  for op <- binary_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(worker(), tensor(), tensor()) :: reference()
    def unquote(nif_name)(_w, _a, _b), do: nif()

    @doc false
    @spec unquote(op)(worker(), tensor(), tensor()) :: tensor()
    def unquote(op)(w, a, b), do: Async.call(unquote(nif_name)(w, a, b))
  end

  # --- Reductions --------------------------------------------------

  axes_keepdims_reduces = [:sum, :mean, :prod, :max, :min, :all, :any, :logsumexp]

  for op <- axes_keepdims_reduces do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(worker(), tensor(), [integer()], boolean()) :: reference()
    def unquote(nif_name)(_w, _a, _axes, _keepdims), do: nif()

    @doc false
    @spec unquote(op)(worker(), tensor(), [integer()], boolean()) :: tensor()
    def unquote(op)(w, a, axes, keepdims),
      do: Async.call(unquote(nif_name)(w, a, axes, keepdims))
  end

  @doc false
  @spec var_nif(worker(), tensor(), [integer()], boolean(), integer()) :: reference()
  def var_nif(_w, _a, _axes, _keepdims, _ddof), do: nif()

  @spec var(worker(), tensor(), [integer()], boolean(), integer()) :: tensor()
  def var(w, a, axes, keepdims, ddof),
    do: Async.call(var_nif(w, a, axes, keepdims, ddof))

  @doc false
  @spec std_nif(worker(), tensor(), [integer()], boolean(), integer()) :: reference()
  def std_nif(_w, _a, _axes, _keepdims, _ddof), do: nif()

  @spec std(worker(), tensor(), [integer()], boolean(), integer()) :: tensor()
  def std(w, a, axes, keepdims, ddof),
    do: Async.call(std_nif(w, a, axes, keepdims, ddof))

  @doc false
  @spec argmax_nif(worker(), tensor(), integer(), boolean()) :: reference()
  def argmax_nif(_w, _a, _axis, _keepdims), do: nif()

  @spec argmax(worker(), tensor(), integer(), boolean()) :: tensor()
  def argmax(w, a, axis, keepdims),
    do: Async.call(argmax_nif(w, a, axis, keepdims))

  @doc false
  @spec argmin_nif(worker(), tensor(), integer(), boolean()) :: reference()
  def argmin_nif(_w, _a, _axis, _keepdims), do: nif()

  @spec argmin(worker(), tensor(), integer(), boolean()) :: tensor()
  def argmin(w, a, axis, keepdims),
    do: Async.call(argmin_nif(w, a, axis, keepdims))

  cumulative_ops = [:cumsum, :cumprod, :cummax, :cummin]

  for op <- cumulative_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(worker(), tensor(), integer(), boolean(), boolean()) ::
            reference()
    def unquote(nif_name)(_w, _a, _axis, _reverse, _inclusive), do: nif()

    @doc false
    @spec unquote(op)(worker(), tensor(), integer(), boolean(), boolean()) :: tensor()
    def unquote(op)(w, a, axis, reverse, inclusive),
      do: Async.call(unquote(nif_name)(w, a, axis, reverse, inclusive))
  end

  # --- Shape -------------------------------------------------------

  @doc false
  @spec reshape_nif(worker(), tensor(), [non_neg_integer()]) :: reference()
  def reshape_nif(_w, _a, _shape), do: nif()

  @spec reshape(worker(), tensor(), [non_neg_integer()]) :: tensor()
  def reshape(w, a, shape), do: Async.call(reshape_nif(w, a, shape))

  @doc false
  @spec transpose_nif(worker(), tensor(), [integer()]) :: reference()
  def transpose_nif(_w, _a, _axes), do: nif()

  @spec transpose(worker(), tensor(), [integer()]) :: tensor()
  def transpose(w, a, axes), do: Async.call(transpose_nif(w, a, axes))

  @doc false
  @spec squeeze_nif(worker(), tensor(), [integer()]) :: reference()
  def squeeze_nif(_w, _a, _axes), do: nif()

  @spec squeeze(worker(), tensor(), [integer()]) :: tensor()
  def squeeze(w, a, axes), do: Async.call(squeeze_nif(w, a, axes))

  @doc false
  @spec expand_dims_nif(worker(), tensor(), [integer()]) :: reference()
  def expand_dims_nif(_w, _a, _axes), do: nif()

  @spec expand_dims(worker(), tensor(), [integer()]) :: tensor()
  def expand_dims(w, a, axes), do: Async.call(expand_dims_nif(w, a, axes))

  @doc false
  @spec broadcast_to_nif(worker(), tensor(), [non_neg_integer()]) :: reference()
  def broadcast_to_nif(_w, _a, _shape), do: nif()

  @spec broadcast_to(worker(), tensor(), [non_neg_integer()]) :: tensor()
  def broadcast_to(w, a, shape), do: Async.call(broadcast_to_nif(w, a, shape))

  @doc false
  @spec concatenate_nif(worker(), [tensor()], integer()) :: reference()
  def concatenate_nif(_w, _arrays, _axis), do: nif()

  @spec concatenate(worker(), [tensor()], integer()) :: tensor()
  def concatenate(w, arrays, axis), do: Async.call(concatenate_nif(w, arrays, axis))

  @doc false
  @spec stack_nif(worker(), [tensor()], integer()) :: reference()
  def stack_nif(_w, _arrays, _axis), do: nif()

  @spec stack(worker(), [tensor()], integer()) :: tensor()
  def stack(w, arrays, axis), do: Async.call(stack_nif(w, arrays, axis))

  @doc false
  @spec flatten_nif(worker(), tensor(), integer(), integer()) :: reference()
  def flatten_nif(_w, _a, _start_axis, _end_axis), do: nif()

  @spec flatten(worker(), tensor(), integer(), integer()) :: tensor()
  def flatten(w, a, start_axis, end_axis),
    do: Async.call(flatten_nif(w, a, start_axis, end_axis))

  @doc false
  @spec tile_nif(worker(), tensor(), [integer()]) :: reference()
  def tile_nif(_w, _a, _reps), do: nif()

  @spec tile(worker(), tensor(), [integer()]) :: tensor()
  def tile(w, a, reps), do: Async.call(tile_nif(w, a, reps))

  @doc false
  @spec swapaxes_nif(worker(), tensor(), integer(), integer()) :: reference()
  def swapaxes_nif(_w, _a, _axis1, _axis2), do: nif()

  @spec swapaxes(worker(), tensor(), integer(), integer()) :: tensor()
  def swapaxes(w, a, axis1, axis2), do: Async.call(swapaxes_nif(w, a, axis1, axis2))

  @doc false
  @spec flip_nif(worker(), tensor(), integer()) :: reference()
  def flip_nif(_w, _a, _axis), do: nif()

  @spec flip(worker(), tensor(), integer()) :: tensor()
  def flip(w, a, axis), do: Async.call(flip_nif(w, a, axis))

  @doc false
  @spec pad_nif(worker(), tensor(), [integer()], [integer()], [integer()], tensor()) ::
          reference()
  def pad_nif(_w, _a, _axes, _low_pad, _high_pad, _pad_value), do: nif()

  @spec pad(worker(), tensor(), [integer()], [integer()], [integer()], tensor()) :: tensor()
  def pad(w, a, axes, low_pad, high_pad, pad_value),
    do: Async.call(pad_nif(w, a, axes, low_pad, high_pad, pad_value))

  @doc false
  @spec repeat_nif(worker(), tensor(), integer(), integer()) :: reference()
  def repeat_nif(_w, _a, _repeats, _axis), do: nif()

  @spec repeat(worker(), tensor(), integer(), integer()) :: tensor()
  def repeat(w, a, repeats, axis), do: Async.call(repeat_nif(w, a, repeats, axis))

  # --- Indexing ----------------------------------------------------

  @doc false
  @spec slice_nif(worker(), tensor(), [integer()], [integer()], [integer()]) :: reference()
  def slice_nif(_w, _a, _start, _stop, _strides), do: nif()

  @spec slice(worker(), tensor(), [integer()], [integer()], [integer()]) :: tensor()
  def slice(w, a, start, stop, strides),
    do: Async.call(slice_nif(w, a, start, stop, strides))

  @doc false
  @spec slice_update_nif(worker(), tensor(), tensor(), [integer()]) :: reference()
  def slice_update_nif(_w, _src, _update, _start), do: nif()

  @spec slice_update(worker(), tensor(), tensor(), [integer()]) :: tensor()
  def slice_update(w, src, update, start),
    do: Async.call(slice_update_nif(w, src, update, start))

  @doc false
  @spec take_nif(worker(), tensor(), tensor(), integer()) :: reference()
  def take_nif(_w, _a, _indices, _axis), do: nif()

  @spec take(worker(), tensor(), tensor(), integer()) :: tensor()
  def take(w, a, indices, axis), do: Async.call(take_nif(w, a, indices, axis))

  @doc false
  @spec where_nif(worker(), tensor(), tensor(), tensor()) :: reference()
  def where_nif(_w, _cond, _x, _y), do: nif()

  @spec where(worker(), tensor(), tensor(), tensor()) :: tensor()
  def where(w, cond, x, y), do: Async.call(where_nif(w, cond, x, y))

  # --- Linalg ------------------------------------------------------

  @doc false
  @spec matmul_nif(worker(), tensor(), tensor()) :: reference()
  def matmul_nif(_w, _a, _b), do: nif()

  @spec matmul(worker(), tensor(), tensor()) :: tensor()
  def matmul(w, a, b), do: Async.call(matmul_nif(w, a, b))

  @doc false
  @spec tensordot_nif(worker(), tensor(), tensor(), [integer()], [integer()]) :: reference()
  def tensordot_nif(_w, _a, _b, _axes_a, _axes_b), do: nif()

  @spec tensordot(worker(), tensor(), tensor(), [integer()], [integer()]) :: tensor()
  def tensordot(w, a, b, axes_a, axes_b),
    do: Async.call(tensordot_nif(w, a, b, axes_a, axes_b))

  @doc false
  @spec outer_nif(worker(), tensor(), tensor()) :: reference()
  def outer_nif(_w, _a, _b), do: nif()

  @spec outer(worker(), tensor(), tensor()) :: tensor()
  def outer(w, a, b), do: Async.call(outer_nif(w, a, b))

  @doc false
  @spec inner_nif(worker(), tensor(), tensor()) :: reference()
  def inner_nif(_w, _a, _b), do: nif()

  @spec inner(worker(), tensor(), tensor()) :: tensor()
  def inner(w, a, b), do: Async.call(inner_nif(w, a, b))

  @doc false
  @spec einsum_nif(worker(), String.t(), [tensor()]) :: reference()
  def einsum_nif(_w, _subscripts, _operands), do: nif()

  @spec einsum(worker(), String.t(), [tensor()]) :: tensor()
  def einsum(w, subscripts, operands),
    do: Async.call(einsum_nif(w, subscripts, operands))

  # --- Linalg (decompositions / solvers) ---------------------------

  @doc false
  @spec linalg_lu_nif(worker(), tensor()) :: reference()
  def linalg_lu_nif(_w, _a), do: nif()

  @spec linalg_lu(worker(), tensor()) :: {tensor(), tensor(), tensor()}
  def linalg_lu(w, a), do: Async.call(linalg_lu_nif(w, a))

  @doc false
  @spec linalg_svd_nif(worker(), tensor()) :: reference()
  def linalg_svd_nif(_w, _a), do: nif()

  @spec linalg_svd(worker(), tensor()) :: {tensor(), tensor(), tensor()}
  def linalg_svd(w, a), do: Async.call(linalg_svd_nif(w, a))

  @doc false
  @spec linalg_qr_nif(worker(), tensor()) :: reference()
  def linalg_qr_nif(_w, _a), do: nif()

  @spec linalg_qr(worker(), tensor()) :: {tensor(), tensor()}
  def linalg_qr(w, a), do: Async.call(linalg_qr_nif(w, a))

  @doc false
  @spec linalg_cholesky_nif(worker(), tensor(), boolean()) :: reference()
  def linalg_cholesky_nif(_w, _a, _upper), do: nif()

  @spec linalg_cholesky(worker(), tensor(), boolean()) :: tensor()
  def linalg_cholesky(w, a, upper), do: Async.call(linalg_cholesky_nif(w, a, upper))

  @doc false
  @spec linalg_eigh_nif(worker(), tensor(), String.t()) :: reference()
  def linalg_eigh_nif(_w, _a, _uplo), do: nif()

  @spec linalg_eigh(worker(), tensor(), String.t()) :: {tensor(), tensor()}
  def linalg_eigh(w, a, uplo), do: Async.call(linalg_eigh_nif(w, a, uplo))

  @doc false
  @spec linalg_solve_nif(worker(), tensor(), tensor()) :: reference()
  def linalg_solve_nif(_w, _a, _b), do: nif()

  @spec linalg_solve(worker(), tensor(), tensor()) :: tensor()
  def linalg_solve(w, a, b), do: Async.call(linalg_solve_nif(w, a, b))

  @doc false
  @spec linalg_solve_triangular_nif(worker(), tensor(), tensor(), boolean()) :: reference()
  def linalg_solve_triangular_nif(_w, _a, _b, _upper), do: nif()

  @spec linalg_solve_triangular(worker(), tensor(), tensor(), boolean()) :: tensor()
  def linalg_solve_triangular(w, a, b, upper),
    do: Async.call(linalg_solve_triangular_nif(w, a, b, upper))

  # --- Quantization ------------------------------------------------

  @doc false
  @spec quantize_nif(worker(), tensor(), integer(), integer(), String.t()) ::
          reference()
  def quantize_nif(_w, _w_tensor, _group_size, _bits, _mode), do: nif()

  # For microscaled modes MLX's `fp_quantize` returns only `(wq, scales)`;
  # the NIF substitutes a scalar-zero placeholder for the third tuple
  # element so the return shape is uniform. The Elixir layer
  # (`QuantizedWeight.to_dense/1`, `Emily.Quantization.quantized_matmul/2`)
  # passes `nil` for biases on non-affine modes — the placeholder is
  # never fed back into MLX.
  @spec quantize(worker(), tensor(), integer(), integer(), String.t()) ::
          {tensor(), tensor(), tensor()}
  def quantize(w, w_tensor, group_size, bits, mode),
    do: Async.call(quantize_nif(w, w_tensor, group_size, bits, mode))

  @doc false
  @spec dequantize_nif(
          worker(),
          tensor(),
          tensor(),
          tensor() | nil,
          integer(),
          integer(),
          String.t()
        ) ::
          reference()
  def dequantize_nif(_w, _w_q, _scales, _biases, _group_size, _bits, _mode),
    do: nif()

  @spec dequantize(
          worker(),
          tensor(),
          tensor(),
          tensor() | nil,
          integer(),
          integer(),
          String.t()
        ) ::
          tensor()
  def dequantize(w, w_q, scales, biases, group_size, bits, mode),
    do: Async.call(dequantize_nif(w, w_q, scales, biases, group_size, bits, mode))

  @doc false
  @spec quantized_matmul_nif(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          tensor() | nil,
          boolean(),
          integer(),
          integer(),
          String.t()
        ) :: reference()
  # credo:disable-for-next-line Credo.Check.Refactor.FunctionArity
  def quantized_matmul_nif(
        _w,
        _x,
        _w_q,
        _scales,
        _biases,
        _transpose,
        _group_size,
        _bits,
        _mode
      ),
      do: nif()

  @spec quantized_matmul(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          tensor() | nil,
          boolean(),
          integer(),
          integer(),
          String.t()
        ) :: tensor()
  # credo:disable-for-next-line Credo.Check.Refactor.FunctionArity
  def quantized_matmul(w, x, w_q, scales, biases, transpose, group_size, bits, mode),
    do:
      Async.call(
        quantized_matmul_nif(w, x, w_q, scales, biases, transpose, group_size, bits, mode)
      )

  # --- Fast / fused transformer kernels ---------------------------

  @doc false
  @spec fast_rms_norm_nif(worker(), tensor(), tensor() | nil, float()) :: reference()
  def fast_rms_norm_nif(_w, _x, _weight, _eps), do: nif()

  @spec fast_rms_norm(worker(), tensor(), tensor() | nil, float()) :: tensor()
  def fast_rms_norm(w, x, weight, eps),
    do: Async.call(fast_rms_norm_nif(w, x, weight, eps))

  @doc false
  @spec fast_layer_norm_nif(worker(), tensor(), tensor() | nil, tensor() | nil, float()) ::
          reference()
  def fast_layer_norm_nif(_w, _x, _weight, _bias, _eps), do: nif()

  @spec fast_layer_norm(worker(), tensor(), tensor() | nil, tensor() | nil, float()) ::
          tensor()
  def fast_layer_norm(w, x, weight, bias, eps),
    do: Async.call(fast_layer_norm_nif(w, x, weight, bias, eps))

  @doc false
  @spec fast_rope_nif(
          worker(),
          tensor(),
          integer(),
          boolean(),
          float() | nil,
          float(),
          tensor(),
          tensor() | nil
        ) :: reference()
  def fast_rope_nif(_w, _x, _dims, _traditional, _base, _scale, _offset, _freqs), do: nif()

  @spec fast_rope(
          worker(),
          tensor(),
          integer(),
          boolean(),
          float() | nil,
          float(),
          tensor(),
          tensor() | nil
        ) :: tensor()
  def fast_rope(w, x, dims, traditional, base, scale, offset, freqs),
    do: Async.call(fast_rope_nif(w, x, dims, traditional, base, scale, offset, freqs))

  @doc false
  @spec fast_scaled_dot_product_attention_nif(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          float(),
          String.t(),
          [tensor()],
          [tensor()]
        ) :: reference()
  def fast_scaled_dot_product_attention_nif(
        _w,
        _q,
        _k,
        _v,
        _scale,
        _mask_mode,
        _mask_arrs,
        _sinks_arrs
      ),
      do: nif()

  @spec fast_scaled_dot_product_attention(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          float(),
          String.t(),
          [tensor()],
          [tensor()]
        ) :: tensor()
  def fast_scaled_dot_product_attention(w, q, k, v, scale, mask_mode, mask_arrs, sinks_arrs),
    do:
      Async.call(
        fast_scaled_dot_product_attention_nif(
          w,
          q,
          k,
          v,
          scale,
          mask_mode,
          mask_arrs,
          sinks_arrs
        )
      )

  # --- Sort --------------------------------------------------------

  @doc false
  @spec sort_nif(worker(), tensor(), integer()) :: reference()
  def sort_nif(_w, _a, _axis), do: nif()

  @spec sort(worker(), tensor(), integer()) :: tensor()
  def sort(w, a, axis), do: Async.call(sort_nif(w, a, axis))

  @doc false
  @spec argsort_nif(worker(), tensor(), integer()) :: reference()
  def argsort_nif(_w, _a, _axis), do: nif()

  @spec argsort(worker(), tensor(), integer()) :: tensor()
  def argsort(w, a, axis), do: Async.call(argsort_nif(w, a, axis))

  @doc false
  @spec partition_nif(worker(), tensor(), integer(), integer()) :: reference()
  def partition_nif(_w, _a, _kth, _axis), do: nif()

  @spec partition(worker(), tensor(), integer(), integer()) :: tensor()
  def partition(w, a, kth, axis), do: Async.call(partition_nif(w, a, kth, axis))

  @doc false
  @spec argpartition_nif(worker(), tensor(), integer(), integer()) :: reference()
  def argpartition_nif(_w, _a, _kth, _axis), do: nif()

  @spec argpartition(worker(), tensor(), integer(), integer()) :: tensor()
  def argpartition(w, a, kth, axis), do: Async.call(argpartition_nif(w, a, kth, axis))

  @doc false
  @spec topk_nif(worker(), tensor(), integer(), integer()) :: reference()
  def topk_nif(_w, _a, _k, _axis), do: nif()

  @spec topk(worker(), tensor(), integer(), integer()) :: tensor()
  def topk(w, a, k, axis), do: Async.call(topk_nif(w, a, k, axis))

  # --- Misc --------------------------------------------------------

  @doc false
  @spec clip_nif(worker(), tensor(), tensor(), tensor()) :: reference()
  def clip_nif(_w, _a, _a_min, _a_max), do: nif()

  @spec clip(worker(), tensor(), tensor(), tensor()) :: tensor()
  def clip(w, a, a_min, a_max), do: Async.call(clip_nif(w, a, a_min, a_max))

  @doc false
  @spec roll_nif(worker(), tensor(), integer(), integer()) :: reference()
  def roll_nif(_w, _a, _shift, _axis), do: nif()

  @spec roll(worker(), tensor(), integer(), integer()) :: tensor()
  def roll(w, a, shift, axis), do: Async.call(roll_nif(w, a, shift, axis))

  @doc false
  @spec softmax_nif(worker(), tensor(), [integer()], boolean()) :: reference()
  def softmax_nif(_w, _a, _axes, _precise), do: nif()

  @spec softmax(worker(), tensor(), [integer()], boolean()) :: tensor()
  def softmax(w, a, axes, precise), do: Async.call(softmax_nif(w, a, axes, precise))

  @doc false
  @spec logcumsumexp_nif(worker(), tensor(), integer(), boolean(), boolean()) ::
          reference()
  def logcumsumexp_nif(_w, _a, _axis, _reverse, _inclusive), do: nif()

  @spec logcumsumexp(worker(), tensor(), integer(), boolean(), boolean()) :: tensor()
  def logcumsumexp(w, a, axis, reverse, inclusive),
    do: Async.call(logcumsumexp_nif(w, a, axis, reverse, inclusive))

  @doc false
  @spec array_equal_nif(worker(), tensor(), tensor(), boolean()) :: reference()
  def array_equal_nif(_w, _a, _b, _equal_nan), do: nif()

  @spec array_equal(worker(), tensor(), tensor(), boolean()) :: tensor()
  def array_equal(w, a, b, equal_nan),
    do: Async.call(array_equal_nif(w, a, b, equal_nan))

  # --- Axis-aligned gather/scatter ---------------------------------

  @doc false
  @spec take_along_axis_nif(worker(), tensor(), tensor(), integer()) :: reference()
  def take_along_axis_nif(_w, _a, _indices, _axis), do: nif()

  @spec take_along_axis(worker(), tensor(), tensor(), integer()) :: tensor()
  def take_along_axis(w, a, indices, axis),
    do: Async.call(take_along_axis_nif(w, a, indices, axis))

  @doc false
  @spec put_along_axis_nif(worker(), tensor(), tensor(), tensor(), integer()) ::
          reference()
  def put_along_axis_nif(_w, _a, _indices, _values, _axis), do: nif()

  @spec put_along_axis(worker(), tensor(), tensor(), tensor(), integer()) :: tensor()
  def put_along_axis(w, a, indices, values, axis),
    do: Async.call(put_along_axis_nif(w, a, indices, values, axis))

  @doc false
  @spec scatter_add_axis_nif(worker(), tensor(), tensor(), tensor(), integer()) ::
          reference()
  def scatter_add_axis_nif(_w, _a, _indices, _values, _axis), do: nif()

  @spec scatter_add_axis(worker(), tensor(), tensor(), tensor(), integer()) :: tensor()
  def scatter_add_axis(w, a, indices, values, axis),
    do: Async.call(scatter_add_axis_nif(w, a, indices, values, axis))

  @doc false
  @spec gather_nif(worker(), tensor(), [tensor()], [integer()], [non_neg_integer()]) ::
          reference()
  def gather_nif(_w, _a, _indices, _axes, _slice_sizes), do: nif()

  @spec gather(worker(), tensor(), [tensor()], [integer()], [non_neg_integer()]) ::
          tensor()
  def gather(w, a, indices, axes, slice_sizes),
    do: Async.call(gather_nif(w, a, indices, axes, slice_sizes))

  @doc false
  @spec scatter_nif(worker(), tensor(), [tensor()], tensor(), [integer()]) :: reference()
  def scatter_nif(_w, _a, _indices, _updates, _axes), do: nif()

  @spec scatter(worker(), tensor(), [tensor()], tensor(), [integer()]) :: tensor()
  def scatter(w, a, indices, updates, axes),
    do: Async.call(scatter_nif(w, a, indices, updates, axes))

  @doc false
  @spec scatter_add_nif(worker(), tensor(), [tensor()], tensor(), [integer()]) ::
          reference()
  def scatter_add_nif(_w, _a, _indices, _updates, _axes), do: nif()

  @spec scatter_add(worker(), tensor(), [tensor()], tensor(), [integer()]) :: tensor()
  def scatter_add(w, a, indices, updates, axes),
    do: Async.call(scatter_add_nif(w, a, indices, updates, axes))

  # --- Window / pooling reductions ---------------------------------

  # Composed via MLX's pad + as_strided + reduce (MLX has no native
  # window_* primitives — mirrors nn/layers/pooling.py). Elixir side
  # resolves :valid/:same padding to {lo, hi} pairs, and supplies the
  # dtype-specific identity (`init_value`) as the padding fill.

  window_ops = [:window_sum, :window_max, :window_min, :window_product]

  for op <- window_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(
            worker(),
            tensor(),
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            tensor()
          ) :: reference()
    def unquote(nif_name)(_w, _t, _window, _strides, _pad_lo, _pad_hi, _dilations, _init),
      do: nif()

    @doc false
    @spec unquote(op)(
            worker(),
            tensor(),
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            tensor()
          ) :: tensor()
    def unquote(op)(w, t, window, strides, pad_lo, pad_hi, dilations, init),
      do: Async.call(unquote(nif_name)(w, t, window, strides, pad_lo, pad_hi, dilations, init))
  end

  # --- Window scatters (MaxPool/MinPool backward) ------------------

  # pad(source) -> as_strided -> argmax-with-last-occurrence-tie-break
  # -> scatter_add into full(init_value) -> slice. No dilation — Nx's
  # scatter variants don't accept :window_dilations.

  window_scatter_ops = [:window_scatter_max, :window_scatter_min]

  for op <- window_scatter_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(
            worker(),
            tensor(),
            tensor(),
            tensor(),
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()]
          ) :: reference()
    def unquote(nif_name)(_w, _t, _source, _init, _window, _strides, _pad_lo, _pad_hi),
      do: nif()

    @doc false
    @spec unquote(op)(
            worker(),
            tensor(),
            tensor(),
            tensor(),
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()],
            [non_neg_integer()]
          ) :: tensor()
    def unquote(op)(w, t, source, init, window, strides, pad_lo, pad_hi),
      do: Async.call(unquote(nif_name)(w, t, source, init, window, strides, pad_lo, pad_hi))
  end

  # --- Convolution -------------------------------------------------

  @doc false
  @spec conv_general_nif(
          worker(),
          tensor(),
          tensor(),
          [integer()],
          {[integer()], [integer()]},
          {[integer()], [integer()]},
          integer(),
          boolean()
        ) :: reference()
  def conv_general_nif(_w, _input, _weight, _stride, _padding, _dilation, _groups, _flip),
    do: nif()

  @spec conv_general(
          worker(),
          tensor(),
          tensor(),
          [integer()],
          {[integer()], [integer()]},
          {[integer()], [integer()]},
          integer(),
          boolean()
        ) :: tensor()
  def conv_general(w, input, weight, stride, padding, dilation, groups, flip),
    do: Async.call(conv_general_nif(w, input, weight, stride, padding, dilation, groups, flip))

  # --- Random ------------------------------------------------------

  # random_key is pure computation — no worker, no stream — stays sync.
  @spec random_key(integer()) :: tensor()
  def random_key(_seed), do: nif()

  @doc false
  @spec random_split_nif(worker(), tensor(), integer()) :: reference()
  def random_split_nif(_w, _key, _num), do: nif()

  @spec random_split(worker(), tensor(), integer()) :: tensor()
  def random_split(w, key, num), do: Async.call(random_split_nif(w, key, num))

  @doc false
  @spec random_uniform_nif(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) :: reference()
  def random_uniform_nif(_w, _low, _high, _shape, _dtype, _key), do: nif()

  @spec random_uniform(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) :: tensor()
  def random_uniform(w, low, high, shape, dtype, key),
    do: Async.call(random_uniform_nif(w, low, high, shape, dtype, key))

  @doc false
  @spec random_normal_nif(
          worker(),
          [non_neg_integer()],
          dtype(),
          float(),
          float(),
          tensor() | nil
        ) :: reference()
  def random_normal_nif(_w, _shape, _dtype, _loc, _scale, _key), do: nif()

  @spec random_normal(worker(), [non_neg_integer()], dtype(), float(), float(), tensor() | nil) ::
          tensor()
  def random_normal(w, shape, dtype, loc, scale, key),
    do: Async.call(random_normal_nif(w, shape, dtype, loc, scale, key))

  @doc false
  @spec random_randint_nif(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) :: reference()
  def random_randint_nif(_w, _low, _high, _shape, _dtype, _key), do: nif()

  @spec random_randint(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) :: tensor()
  def random_randint(w, low, high, shape, dtype, key),
    do: Async.call(random_randint_nif(w, low, high, shape, dtype, key))

  @doc false
  @spec random_bernoulli_nif(worker(), tensor(), [non_neg_integer()], tensor() | nil) ::
          reference()
  def random_bernoulli_nif(_w, _p, _shape, _key), do: nif()

  @spec random_bernoulli(worker(), tensor(), [non_neg_integer()], tensor() | nil) :: tensor()
  def random_bernoulli(w, p, shape, key),
    do: Async.call(random_bernoulli_nif(w, p, shape, key))

  @doc false
  @spec random_gumbel_nif(worker(), [non_neg_integer()], dtype(), tensor() | nil) ::
          reference()
  def random_gumbel_nif(_w, _shape, _dtype, _key), do: nif()

  @spec random_gumbel(worker(), [non_neg_integer()], dtype(), tensor() | nil) :: tensor()
  def random_gumbel(w, shape, dtype, key),
    do: Async.call(random_gumbel_nif(w, shape, dtype, key))

  @doc false
  @spec random_categorical_nif(worker(), tensor(), integer(), integer(), tensor() | nil) ::
          reference()
  def random_categorical_nif(_w, _logits, _axis, _num_samples, _key), do: nif()

  @spec random_categorical(worker(), tensor(), integer(), integer(), tensor() | nil) ::
          tensor()
  def random_categorical(w, logits, axis, num_samples, key),
    do: Async.call(random_categorical_nif(w, logits, axis, num_samples, key))

  # --- FFT ---------------------------------------------------------

  fft_ops = [:fftn, :ifftn, :rfftn, :irfftn]

  for op <- fft_ops do
    nif_name = :"#{op}_nif"

    @doc false
    @spec unquote(nif_name)(worker(), tensor(), [non_neg_integer()], [integer()]) ::
            reference()
    def unquote(nif_name)(_w, _a, _n, _axes), do: nif()

    @doc false
    @spec unquote(op)(worker(), tensor(), [non_neg_integer()], [integer()]) :: tensor()
    def unquote(op)(w, a, n, axes), do: Async.call(unquote(nif_name)(w, a, n, axes))
  end

  # --- Memory / allocator ------------------------------------------

  @spec get_active_memory() :: non_neg_integer()
  def get_active_memory, do: nif()

  @spec get_peak_memory() :: non_neg_integer()
  def get_peak_memory, do: nif()

  @spec reset_peak_memory() :: :ok
  def reset_peak_memory, do: nif()

  @spec get_cache_memory() :: non_neg_integer()
  def get_cache_memory, do: nif()

  @spec clear_cache() :: :ok
  def clear_cache, do: nif()
end
