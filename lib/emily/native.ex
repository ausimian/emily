defmodule Emily.Native do
  @moduledoc false
  # Thin NIF loader for the emily C++ shim. Every function here maps
  # directly to one NIF in c_src/. No policy, no caching, no defaults —
  # higher layers do that.
  #
  # Op NIFs take a trailing `stream_index` parameter (int). -1 means
  # "use the default stream" (resolve_stream in tensor.hpp). Core NIFs
  # (from_binary, to_binary, shape, dtype, eval) and memory introspection
  # NIFs don't take a stream — they don't create lazy graph primitives.

  @on_load :__on_load__
  @compile {:autoload, false}

  @doc false
  def __on_load__ do
    path = :filename.join(:code.priv_dir(:emily), ~c"libemily")
    :erlang.load_nif(path, 0)
  end

  @type tensor :: reference()
  @type dtype :: {atom(), non_neg_integer()}

  defp nif, do: :erlang.nif_error(:nif_not_loaded)

  # --- Core --------------------------------------------------------

  @spec from_binary(binary(), [non_neg_integer()], dtype()) :: tensor()
  def from_binary(_data, _shape, _dtype), do: nif()

  @spec to_binary(tensor(), integer()) :: binary()
  def to_binary(_tensor, _s), do: nif()

  @spec shape(tensor()) :: [non_neg_integer()]
  def shape(_tensor), do: nif()

  @spec dtype(tensor()) :: dtype()
  def dtype(_tensor), do: nif()

  @spec eval(tensor()) :: :ok
  def eval(_tensor), do: nif()

  # --- Streams -----------------------------------------------------

  @spec new_stream(atom()) :: integer()
  def new_stream(_device), do: nif()

  @spec set_default_stream(integer()) :: :ok
  def set_default_stream(_stream_index), do: nif()

  @spec get_default_stream(atom()) :: integer()
  def get_default_stream(_device), do: nif()

  @spec synchronize_stream(integer()) :: :ok
  def synchronize_stream(_stream_index), do: nif()

  # --- Creation ----------------------------------------------------

  @spec zeros([non_neg_integer()], dtype(), integer()) :: tensor()
  def zeros(_shape, _dtype, _s), do: nif()

  @spec ones([non_neg_integer()], dtype(), integer()) :: tensor()
  def ones(_shape, _dtype, _s), do: nif()

  @spec full([non_neg_integer()], tensor(), dtype(), integer()) :: tensor()
  def full(_shape, _value, _dtype, _s), do: nif()

  @spec arange(float(), float(), float(), dtype(), integer()) :: tensor()
  def arange(_start, _stop, _step, _dtype, _s), do: nif()

  @spec eye(integer(), integer(), integer(), dtype(), integer()) :: tensor()
  def eye(_n, _m, _k, _dtype, _s), do: nif()

  # --- Cast --------------------------------------------------------

  @spec astype(tensor(), dtype(), integer()) :: tensor()
  def astype(_a, _dtype, _s), do: nif()

  @spec bitcast(tensor(), dtype(), integer()) :: tensor()
  def bitcast(_a, _dtype, _s), do: nif()

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
    @spec unquote(op)(tensor(), integer()) :: tensor()
    def unquote(op)(_a, _s), do: nif()
  end

  @spec round(tensor(), integer(), integer()) :: tensor()
  def round(_a, _decimals, _s), do: nif()

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
    @spec unquote(op)(tensor(), tensor(), integer()) :: tensor()
    def unquote(op)(_a, _b, _s), do: nif()
  end

  # --- Reductions --------------------------------------------------

  axes_keepdims_reduces = [:sum, :mean, :prod, :max, :min, :all, :any, :logsumexp]

  for op <- axes_keepdims_reduces do
    @doc false
    @spec unquote(op)(tensor(), [integer()], boolean(), integer()) :: tensor()
    def unquote(op)(_a, _axes, _keepdims, _s), do: nif()
  end

  @spec var(tensor(), [integer()], boolean(), integer(), integer()) :: tensor()
  def var(_a, _axes, _keepdims, _ddof, _s), do: nif()

  @spec std(tensor(), [integer()], boolean(), integer(), integer()) :: tensor()
  def std(_a, _axes, _keepdims, _ddof, _s), do: nif()

  @spec argmax(tensor(), integer(), boolean(), integer()) :: tensor()
  def argmax(_a, _axis, _keepdims, _s), do: nif()

  @spec argmin(tensor(), integer(), boolean(), integer()) :: tensor()
  def argmin(_a, _axis, _keepdims, _s), do: nif()

  cumulative_ops = [:cumsum, :cumprod, :cummax, :cummin]

  for op <- cumulative_ops do
    @doc false
    @spec unquote(op)(tensor(), integer(), boolean(), boolean(), integer()) :: tensor()
    def unquote(op)(_a, _axis, _reverse, _inclusive, _s), do: nif()
  end

  # --- Shape -------------------------------------------------------

  @spec reshape(tensor(), [non_neg_integer()], integer()) :: tensor()
  def reshape(_a, _shape, _s), do: nif()

  @spec transpose(tensor(), [integer()], integer()) :: tensor()
  def transpose(_a, _axes, _s), do: nif()

  @spec squeeze(tensor(), [integer()], integer()) :: tensor()
  def squeeze(_a, _axes, _s), do: nif()

  @spec expand_dims(tensor(), [integer()], integer()) :: tensor()
  def expand_dims(_a, _axes, _s), do: nif()

  @spec broadcast_to(tensor(), [non_neg_integer()], integer()) :: tensor()
  def broadcast_to(_a, _shape, _s), do: nif()

  @spec concatenate([tensor()], integer(), integer()) :: tensor()
  def concatenate(_arrays, _axis, _s), do: nif()

  @spec stack([tensor()], integer(), integer()) :: tensor()
  def stack(_arrays, _axis, _s), do: nif()

  @spec flatten(tensor(), integer(), integer(), integer()) :: tensor()
  def flatten(_a, _start_axis, _end_axis, _s), do: nif()

  @spec tile(tensor(), [integer()], integer()) :: tensor()
  def tile(_a, _reps, _s), do: nif()

  @spec swapaxes(tensor(), integer(), integer(), integer()) :: tensor()
  def swapaxes(_a, _axis1, _axis2, _s), do: nif()

  @spec pad(tensor(), [integer()], [integer()], [integer()], tensor(), integer()) :: tensor()
  def pad(_a, _axes, _low_pad, _high_pad, _pad_value, _s), do: nif()

  @spec repeat(tensor(), integer(), integer(), integer()) :: tensor()
  def repeat(_a, _repeats, _axis, _s), do: nif()

  # --- Indexing ----------------------------------------------------

  @spec slice(tensor(), [integer()], [integer()], [integer()], integer()) :: tensor()
  def slice(_a, _start, _stop, _strides, _s), do: nif()

  @spec slice_update(tensor(), tensor(), [integer()], integer()) :: tensor()
  def slice_update(_src, _update, _start, _s), do: nif()

  @spec take(tensor(), tensor(), integer(), integer()) :: tensor()
  def take(_a, _indices, _axis, _s), do: nif()

  @spec where(tensor(), tensor(), tensor(), integer()) :: tensor()
  def where(_cond, _x, _y, _s), do: nif()

  # --- Linalg ------------------------------------------------------

  @spec matmul(tensor(), tensor(), integer()) :: tensor()
  def matmul(_a, _b, _s), do: nif()

  @spec tensordot(tensor(), tensor(), [integer()], [integer()], integer()) :: tensor()
  def tensordot(_a, _b, _axes_a, _axes_b, _s), do: nif()

  @spec outer(tensor(), tensor(), integer()) :: tensor()
  def outer(_a, _b, _s), do: nif()

  @spec inner(tensor(), tensor(), integer()) :: tensor()
  def inner(_a, _b, _s), do: nif()

  # --- Quantization ------------------------------------------------

  @spec quantize(tensor(), integer(), integer(), integer()) ::
          {tensor(), tensor(), tensor()}
  def quantize(_w, _group_size, _bits, _s), do: nif()

  @spec dequantize(tensor(), tensor(), tensor(), integer(), integer(), integer()) ::
          tensor()
  def dequantize(_w_q, _scales, _biases, _group_size, _bits, _s), do: nif()

  @spec quantized_matmul(
          tensor(),
          tensor(),
          tensor(),
          tensor(),
          boolean(),
          integer(),
          integer(),
          integer()
        ) :: tensor()
  def quantized_matmul(
        _x,
        _w_q,
        _scales,
        _biases,
        _transpose,
        _group_size,
        _bits,
        _s
      ),
      do: nif()

  # --- Fast / fused transformer kernels ---------------------------

  @spec fast_rms_norm(tensor(), tensor() | nil, float(), integer()) :: tensor()
  def fast_rms_norm(_x, _weight, _eps, _s), do: nif()

  @spec fast_layer_norm(tensor(), tensor() | nil, tensor() | nil, float(), integer()) :: tensor()
  def fast_layer_norm(_x, _weight, _bias, _eps, _s), do: nif()

  @spec fast_rope(
          tensor(),
          integer(),
          boolean(),
          float() | nil,
          float(),
          tensor(),
          tensor() | nil,
          integer()
        ) :: tensor()
  def fast_rope(_x, _dims, _traditional, _base, _scale, _offset, _freqs, _s), do: nif()

  @spec fast_scaled_dot_product_attention(
          tensor(),
          tensor(),
          tensor(),
          float(),
          String.t(),
          [tensor()],
          integer()
        ) :: tensor()
  def fast_scaled_dot_product_attention(_q, _k, _v, _scale, _mask_mode, _mask_arrs, _s),
    do: nif()

  # --- Sort --------------------------------------------------------

  @spec sort(tensor(), integer(), integer()) :: tensor()
  def sort(_a, _axis, _s), do: nif()

  @spec argsort(tensor(), integer(), integer()) :: tensor()
  def argsort(_a, _axis, _s), do: nif()

  @spec partition(tensor(), integer(), integer(), integer()) :: tensor()
  def partition(_a, _kth, _axis, _s), do: nif()

  @spec argpartition(tensor(), integer(), integer(), integer()) :: tensor()
  def argpartition(_a, _kth, _axis, _s), do: nif()

  @spec topk(tensor(), integer(), integer(), integer()) :: tensor()
  def topk(_a, _k, _axis, _s), do: nif()

  # --- Misc --------------------------------------------------------

  @spec clip(tensor(), tensor(), tensor(), integer()) :: tensor()
  def clip(_a, _a_min, _a_max, _s), do: nif()

  @spec roll(tensor(), integer(), integer(), integer()) :: tensor()
  def roll(_a, _shift, _axis, _s), do: nif()

  @spec softmax(tensor(), [integer()], boolean(), integer()) :: tensor()
  def softmax(_a, _axes, _precise, _s), do: nif()

  @spec logcumsumexp(tensor(), integer(), boolean(), boolean(), integer()) :: tensor()
  def logcumsumexp(_a, _axis, _reverse, _inclusive, _s), do: nif()

  @spec array_equal(tensor(), tensor(), boolean(), integer()) :: tensor()
  def array_equal(_a, _b, _equal_nan, _s), do: nif()

  # --- Axis-aligned gather/scatter ---------------------------------

  @spec take_along_axis(tensor(), tensor(), integer(), integer()) :: tensor()
  def take_along_axis(_a, _indices, _axis, _s), do: nif()

  @spec put_along_axis(tensor(), tensor(), tensor(), integer(), integer()) :: tensor()
  def put_along_axis(_a, _indices, _values, _axis, _s), do: nif()

  @spec scatter_add_axis(tensor(), tensor(), tensor(), integer(), integer()) :: tensor()
  def scatter_add_axis(_a, _indices, _values, _axis, _s), do: nif()

  @spec gather(tensor(), [tensor()], [integer()], [non_neg_integer()], integer()) :: tensor()
  def gather(_a, _indices, _axes, _slice_sizes, _s), do: nif()

  @spec scatter(tensor(), [tensor()], tensor(), [integer()], integer()) :: tensor()
  def scatter(_a, _indices, _updates, _axes, _s), do: nif()

  @spec scatter_add(tensor(), [tensor()], tensor(), [integer()], integer()) :: tensor()
  def scatter_add(_a, _indices, _updates, _axes, _s), do: nif()

  # --- Convolution -------------------------------------------------

  @spec conv_general(
          tensor(),
          tensor(),
          [integer()],
          {[integer()], [integer()]},
          {[integer()], [integer()]},
          integer(),
          boolean(),
          integer()
        ) :: tensor()
  def conv_general(
        _input,
        _weight,
        _stride,
        _padding,
        _dilation,
        _groups,
        _flip,
        _s
      ),
      do: nif()

  # --- Random ------------------------------------------------------

  @spec random_key(integer()) :: tensor()
  def random_key(_seed), do: nif()

  @spec random_split(tensor(), integer(), integer()) :: tensor()
  def random_split(_key, _num, _s), do: nif()

  @spec random_uniform(
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil,
          integer()
        ) ::
          tensor()
  def random_uniform(_low, _high, _shape, _dtype, _key, _s), do: nif()

  @spec random_normal([non_neg_integer()], dtype(), float(), float(), tensor() | nil, integer()) ::
          tensor()
  def random_normal(_shape, _dtype, _loc, _scale, _key, _s), do: nif()

  @spec random_randint(
          tensor(),
          tensor(),
          [non_neg_integer()],
          dtype(),
          tensor() | nil,
          integer()
        ) ::
          tensor()
  def random_randint(_low, _high, _shape, _dtype, _key, _s), do: nif()

  @spec random_bernoulli(tensor(), [non_neg_integer()], tensor() | nil, integer()) :: tensor()
  def random_bernoulli(_p, _shape, _key, _s), do: nif()

  @spec random_gumbel([non_neg_integer()], dtype(), tensor() | nil, integer()) :: tensor()
  def random_gumbel(_shape, _dtype, _key, _s), do: nif()

  @spec random_categorical(tensor(), integer(), integer(), tensor() | nil, integer()) :: tensor()
  def random_categorical(_logits, _axis, _num_samples, _key, _s), do: nif()

  # --- FFT ---------------------------------------------------------

  @spec fftn(tensor(), [non_neg_integer()], [integer()], integer()) :: tensor()
  def fftn(_a, _n, _axes, _s), do: nif()

  @spec ifftn(tensor(), [non_neg_integer()], [integer()], integer()) :: tensor()
  def ifftn(_a, _n, _axes, _s), do: nif()

  @spec rfftn(tensor(), [non_neg_integer()], [integer()], integer()) :: tensor()
  def rfftn(_a, _n, _axes, _s), do: nif()

  @spec irfftn(tensor(), [non_neg_integer()], [integer()], integer()) :: tensor()
  def irfftn(_a, _n, _axes, _s), do: nif()

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
