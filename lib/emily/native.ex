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

  @on_load :__on_load__
  @compile {:autoload, false}

  @doc false
  def __on_load__ do
    path = :filename.join(:code.priv_dir(:emily), ~c"libemily")
    :erlang.load_nif(path, 0)
  end

  @type tensor :: reference()
  @type worker :: reference()
  @type dtype :: {atom(), non_neg_integer()}

  defp nif, do: :erlang.nif_error(:nif_not_loaded)

  # --- Core --------------------------------------------------------

  @spec from_binary(binary(), [non_neg_integer()], dtype()) :: tensor()
  def from_binary(_data, _shape, _dtype), do: nif()

  @spec to_binary(worker(), tensor()) :: binary()
  def to_binary(_w, _tensor), do: nif()

  @spec shape(tensor()) :: [non_neg_integer()]
  def shape(_tensor), do: nif()

  @spec dtype(tensor()) :: dtype()
  def dtype(_tensor), do: nif()

  @spec eval(worker(), tensor()) :: :ok
  def eval(_w, _tensor), do: nif()

  # --- Worker ------------------------------------------------------

  @spec create_worker() :: worker()
  def create_worker, do: nif()

  # --- Creation ----------------------------------------------------

  @spec zeros(worker(), [non_neg_integer()], dtype()) :: tensor()
  def zeros(_w, _shape, _dtype), do: nif()

  @spec ones(worker(), [non_neg_integer()], dtype()) :: tensor()
  def ones(_w, _shape, _dtype), do: nif()

  @spec full(worker(), [non_neg_integer()], tensor(), dtype()) :: tensor()
  def full(_w, _shape, _value, _dtype), do: nif()

  @spec arange(worker(), float(), float(), float(), dtype()) :: tensor()
  def arange(_w, _start, _stop, _step, _dtype), do: nif()

  @spec eye(worker(), integer(), integer(), integer(), dtype()) :: tensor()
  def eye(_w, _n, _m, _k, _dtype), do: nif()

  # --- Cast --------------------------------------------------------

  @spec astype(worker(), tensor(), dtype()) :: tensor()
  def astype(_w, _a, _dtype), do: nif()

  @spec bitcast(worker(), tensor(), dtype()) :: tensor()
  def bitcast(_w, _a, _dtype), do: nif()

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
    @doc false
    @spec unquote(op)(worker(), tensor()) :: tensor()
    def unquote(op)(_w, _a), do: nif()
  end

  @spec round(worker(), tensor(), integer()) :: tensor()
  def round(_w, _a, _decimals), do: nif()

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
    @doc false
    @spec unquote(op)(worker(), tensor(), tensor()) :: tensor()
    def unquote(op)(_w, _a, _b), do: nif()
  end

  # --- Reductions --------------------------------------------------

  axes_keepdims_reduces = [:sum, :mean, :prod, :max, :min, :all, :any, :logsumexp]

  for op <- axes_keepdims_reduces do
    @doc false
    @spec unquote(op)(worker(), tensor(), [integer()], boolean()) :: tensor()
    def unquote(op)(_w, _a, _axes, _keepdims), do: nif()
  end

  @spec var(worker(), tensor(), [integer()], boolean(), integer()) :: tensor()
  def var(_w, _a, _axes, _keepdims, _ddof), do: nif()

  @spec std(worker(), tensor(), [integer()], boolean(), integer()) :: tensor()
  def std(_w, _a, _axes, _keepdims, _ddof), do: nif()

  @spec argmax(worker(), tensor(), integer(), boolean()) :: tensor()
  def argmax(_w, _a, _axis, _keepdims), do: nif()

  @spec argmin(worker(), tensor(), integer(), boolean()) :: tensor()
  def argmin(_w, _a, _axis, _keepdims), do: nif()

  cumulative_ops = [:cumsum, :cumprod, :cummax, :cummin]

  for op <- cumulative_ops do
    @doc false
    @spec unquote(op)(worker(), tensor(), integer(), boolean(), boolean()) :: tensor()
    def unquote(op)(_w, _a, _axis, _reverse, _inclusive), do: nif()
  end

  # --- Shape -------------------------------------------------------

  @spec reshape(worker(), tensor(), [non_neg_integer()]) :: tensor()
  def reshape(_w, _a, _shape), do: nif()

  @spec transpose(worker(), tensor(), [integer()]) :: tensor()
  def transpose(_w, _a, _axes), do: nif()

  @spec squeeze(worker(), tensor(), [integer()]) :: tensor()
  def squeeze(_w, _a, _axes), do: nif()

  @spec expand_dims(worker(), tensor(), [integer()]) :: tensor()
  def expand_dims(_w, _a, _axes), do: nif()

  @spec broadcast_to(worker(), tensor(), [non_neg_integer()]) :: tensor()
  def broadcast_to(_w, _a, _shape), do: nif()

  @spec concatenate(worker(), [tensor()], integer()) :: tensor()
  def concatenate(_w, _arrays, _axis), do: nif()

  @spec stack(worker(), [tensor()], integer()) :: tensor()
  def stack(_w, _arrays, _axis), do: nif()

  @spec flatten(worker(), tensor(), integer(), integer()) :: tensor()
  def flatten(_w, _a, _start_axis, _end_axis), do: nif()

  @spec tile(worker(), tensor(), [integer()]) :: tensor()
  def tile(_w, _a, _reps), do: nif()

  @spec swapaxes(worker(), tensor(), integer(), integer()) :: tensor()
  def swapaxes(_w, _a, _axis1, _axis2), do: nif()

  @spec pad(worker(), tensor(), [integer()], [integer()], [integer()], tensor()) :: tensor()
  def pad(_w, _a, _axes, _low_pad, _high_pad, _pad_value), do: nif()

  @spec repeat(worker(), tensor(), integer(), integer()) :: tensor()
  def repeat(_w, _a, _repeats, _axis), do: nif()

  # --- Indexing ----------------------------------------------------

  @spec slice(worker(), tensor(), [integer()], [integer()], [integer()]) :: tensor()
  def slice(_w, _a, _start, _stop, _strides), do: nif()

  @spec slice_update(worker(), tensor(), tensor(), [integer()]) :: tensor()
  def slice_update(_w, _src, _update, _start), do: nif()

  @spec take(worker(), tensor(), tensor(), integer()) :: tensor()
  def take(_w, _a, _indices, _axis), do: nif()

  @spec where(worker(), tensor(), tensor(), tensor()) :: tensor()
  def where(_w, _cond, _x, _y), do: nif()

  # --- Linalg ------------------------------------------------------

  @spec matmul(worker(), tensor(), tensor()) :: tensor()
  def matmul(_w, _a, _b), do: nif()

  @spec tensordot(worker(), tensor(), tensor(), [integer()], [integer()]) :: tensor()
  def tensordot(_w, _a, _b, _axes_a, _axes_b), do: nif()

  @spec outer(worker(), tensor(), tensor()) :: tensor()
  def outer(_w, _a, _b), do: nif()

  @spec inner(worker(), tensor(), tensor()) :: tensor()
  def inner(_w, _a, _b), do: nif()

  # --- Quantization ------------------------------------------------

  @spec quantize(worker(), tensor(), integer(), integer()) ::
          {tensor(), tensor(), tensor()}
  def quantize(_w, _w_tensor, _group_size, _bits), do: nif()

  @spec dequantize(worker(), tensor(), tensor(), tensor(), integer(), integer()) ::
          tensor()
  def dequantize(_w, _w_q, _scales, _biases, _group_size, _bits), do: nif()

  @spec quantized_matmul(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          tensor(),
          boolean(),
          integer(),
          integer()
        ) :: tensor()
  def quantized_matmul(
        _w,
        _x,
        _w_q,
        _scales,
        _biases,
        _transpose,
        _group_size,
        _bits
      ),
      do: nif()

  # --- Fast / fused transformer kernels ---------------------------

  @spec fast_rms_norm(worker(), tensor(), tensor() | nil, float()) :: tensor()
  def fast_rms_norm(_w, _x, _weight, _eps), do: nif()

  @spec fast_layer_norm(worker(), tensor(), tensor() | nil, tensor() | nil, float()) :: tensor()
  def fast_layer_norm(_w, _x, _weight, _bias, _eps), do: nif()

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
  def fast_rope(_w, _x, _dims, _traditional, _base, _scale, _offset, _freqs), do: nif()

  @spec fast_scaled_dot_product_attention(
          worker(),
          tensor(),
          tensor(),
          tensor(),
          float(),
          String.t(),
          [tensor()]
        ) :: tensor()
  def fast_scaled_dot_product_attention(_w, _q, _k, _v, _scale, _mask_mode, _mask_arrs),
    do: nif()

  # --- Sort --------------------------------------------------------

  @spec sort(worker(), tensor(), integer()) :: tensor()
  def sort(_w, _a, _axis), do: nif()

  @spec argsort(worker(), tensor(), integer()) :: tensor()
  def argsort(_w, _a, _axis), do: nif()

  @spec partition(worker(), tensor(), integer(), integer()) :: tensor()
  def partition(_w, _a, _kth, _axis), do: nif()

  @spec argpartition(worker(), tensor(), integer(), integer()) :: tensor()
  def argpartition(_w, _a, _kth, _axis), do: nif()

  @spec topk(worker(), tensor(), integer(), integer()) :: tensor()
  def topk(_w, _a, _k, _axis), do: nif()

  # --- Misc --------------------------------------------------------

  @spec clip(worker(), tensor(), tensor(), tensor()) :: tensor()
  def clip(_w, _a, _a_min, _a_max), do: nif()

  @spec roll(worker(), tensor(), integer(), integer()) :: tensor()
  def roll(_w, _a, _shift, _axis), do: nif()

  @spec softmax(worker(), tensor(), [integer()], boolean()) :: tensor()
  def softmax(_w, _a, _axes, _precise), do: nif()

  @spec logcumsumexp(worker(), tensor(), integer(), boolean(), boolean()) :: tensor()
  def logcumsumexp(_w, _a, _axis, _reverse, _inclusive), do: nif()

  @spec array_equal(worker(), tensor(), tensor(), boolean()) :: tensor()
  def array_equal(_w, _a, _b, _equal_nan), do: nif()

  # --- Axis-aligned gather/scatter ---------------------------------

  @spec take_along_axis(worker(), tensor(), tensor(), integer()) :: tensor()
  def take_along_axis(_w, _a, _indices, _axis), do: nif()

  @spec put_along_axis(worker(), tensor(), tensor(), tensor(), integer()) :: tensor()
  def put_along_axis(_w, _a, _indices, _values, _axis), do: nif()

  @spec scatter_add_axis(worker(), tensor(), tensor(), tensor(), integer()) :: tensor()
  def scatter_add_axis(_w, _a, _indices, _values, _axis), do: nif()

  @spec gather(worker(), tensor(), [tensor()], [integer()], [non_neg_integer()]) :: tensor()
  def gather(_w, _a, _indices, _axes, _slice_sizes), do: nif()

  @spec scatter(worker(), tensor(), [tensor()], tensor(), [integer()]) :: tensor()
  def scatter(_w, _a, _indices, _updates, _axes), do: nif()

  @spec scatter_add(worker(), tensor(), [tensor()], tensor(), [integer()]) :: tensor()
  def scatter_add(_w, _a, _indices, _updates, _axes), do: nif()

  # --- Convolution -------------------------------------------------

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
  def conv_general(
        _w,
        _input,
        _weight,
        _stride,
        _padding,
        _dilation,
        _groups,
        _flip
      ),
      do: nif()

  # --- Random ------------------------------------------------------

  @spec random_key(integer()) :: tensor()
  def random_key(_seed), do: nif()

  @spec random_split(worker(), tensor(), integer()) :: tensor()
  def random_split(_w, _key, _num), do: nif()

  @spec random_uniform(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) ::
          tensor()
  def random_uniform(_w, _low, _high, _shape, _dtype, _key), do: nif()

  @spec random_normal(worker(), [non_neg_integer()], dtype(), float(), float(), tensor() | nil) ::
          tensor()
  def random_normal(_w, _shape, _dtype, _loc, _scale, _key), do: nif()

  @spec random_randint(
          worker(),
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil
        ) ::
          tensor()
  def random_randint(_w, _low, _high, _shape, _dtype, _key), do: nif()

  @spec random_bernoulli(worker(), tensor(), [non_neg_integer()], tensor() | nil) :: tensor()
  def random_bernoulli(_w, _p, _shape, _key), do: nif()

  @spec random_gumbel(worker(), [non_neg_integer()], dtype(), tensor() | nil) :: tensor()
  def random_gumbel(_w, _shape, _dtype, _key), do: nif()

  @spec random_categorical(worker(), tensor(), integer(), integer(), tensor() | nil) :: tensor()
  def random_categorical(_w, _logits, _axis, _num_samples, _key), do: nif()

  # --- FFT ---------------------------------------------------------

  @spec fftn(worker(), tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def fftn(_w, _a, _n, _axes), do: nif()

  @spec ifftn(worker(), tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def ifftn(_w, _a, _n, _axes), do: nif()

  @spec rfftn(worker(), tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def rfftn(_w, _a, _n, _axes), do: nif()

  @spec irfftn(worker(), tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def irfftn(_w, _a, _n, _axes), do: nif()

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
