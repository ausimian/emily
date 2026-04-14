defmodule Emily.Native do
  @moduledoc false
  # Thin NIF loader for the emily C++ shim. Every function here maps
  # directly to one NIF in c_src/. No policy, no caching, no defaults —
  # higher layers do that.

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

  @spec to_binary(tensor()) :: binary()
  def to_binary(_tensor), do: nif()

  @spec shape(tensor()) :: [non_neg_integer()]
  def shape(_tensor), do: nif()

  @spec dtype(tensor()) :: dtype()
  def dtype(_tensor), do: nif()

  @spec eval(tensor()) :: :ok
  def eval(_tensor), do: nif()

  # --- Creation ----------------------------------------------------

  @spec zeros([non_neg_integer()], dtype()) :: tensor()
  def zeros(_shape, _dtype), do: nif()

  @spec ones([non_neg_integer()], dtype()) :: tensor()
  def ones(_shape, _dtype), do: nif()

  @spec full([non_neg_integer()], tensor(), dtype()) :: tensor()
  def full(_shape, _value, _dtype), do: nif()

  @spec arange(float(), float(), float(), dtype()) :: tensor()
  def arange(_start, _stop, _step, _dtype), do: nif()

  @spec eye(integer(), integer(), integer(), dtype()) :: tensor()
  def eye(_n, _m, _k, _dtype), do: nif()

  # --- Cast --------------------------------------------------------

  @spec astype(tensor(), dtype()) :: tensor()
  def astype(_a, _dtype), do: nif()

  @spec bitcast(tensor(), dtype()) :: tensor()
  def bitcast(_a, _dtype), do: nif()

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
    @spec unquote(op)(tensor()) :: tensor()
    def unquote(op)(_a), do: nif()
  end

  @spec round(tensor(), integer()) :: tensor()
  def round(_a, _decimals), do: nif()

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
    @spec unquote(op)(tensor(), tensor()) :: tensor()
    def unquote(op)(_a, _b), do: nif()
  end

  # --- Reductions --------------------------------------------------

  axes_keepdims_reduces = [:sum, :mean, :prod, :max, :min, :all, :any, :logsumexp]

  for op <- axes_keepdims_reduces do
    @doc false
    @spec unquote(op)(tensor(), [integer()], boolean()) :: tensor()
    def unquote(op)(_a, _axes, _keepdims), do: nif()
  end

  @spec var(tensor(), [integer()], boolean(), integer()) :: tensor()
  def var(_a, _axes, _keepdims, _ddof), do: nif()

  @spec std(tensor(), [integer()], boolean(), integer()) :: tensor()
  def std(_a, _axes, _keepdims, _ddof), do: nif()

  @spec argmax(tensor(), integer(), boolean()) :: tensor()
  def argmax(_a, _axis, _keepdims), do: nif()

  @spec argmin(tensor(), integer(), boolean()) :: tensor()
  def argmin(_a, _axis, _keepdims), do: nif()

  cumulative_ops = [:cumsum, :cumprod, :cummax, :cummin]

  for op <- cumulative_ops do
    @doc false
    @spec unquote(op)(tensor(), integer(), boolean(), boolean()) :: tensor()
    def unquote(op)(_a, _axis, _reverse, _inclusive), do: nif()
  end

  # --- Shape -------------------------------------------------------

  @spec reshape(tensor(), [non_neg_integer()]) :: tensor()
  def reshape(_a, _shape), do: nif()

  @spec transpose(tensor(), [integer()]) :: tensor()
  def transpose(_a, _axes), do: nif()

  @spec squeeze(tensor(), [integer()]) :: tensor()
  def squeeze(_a, _axes), do: nif()

  @spec expand_dims(tensor(), [integer()]) :: tensor()
  def expand_dims(_a, _axes), do: nif()

  @spec broadcast_to(tensor(), [non_neg_integer()]) :: tensor()
  def broadcast_to(_a, _shape), do: nif()

  @spec concatenate([tensor()], integer()) :: tensor()
  def concatenate(_arrays, _axis), do: nif()

  @spec stack([tensor()], integer()) :: tensor()
  def stack(_arrays, _axis), do: nif()

  @spec flatten(tensor(), integer(), integer()) :: tensor()
  def flatten(_a, _start_axis, _end_axis), do: nif()

  @spec tile(tensor(), [integer()]) :: tensor()
  def tile(_a, _reps), do: nif()

  @spec swapaxes(tensor(), integer(), integer()) :: tensor()
  def swapaxes(_a, _axis1, _axis2), do: nif()

  @spec pad(tensor(), [integer()], [integer()], [integer()], tensor()) :: tensor()
  def pad(_a, _axes, _low_pad, _high_pad, _pad_value), do: nif()

  @spec repeat(tensor(), integer(), integer()) :: tensor()
  def repeat(_a, _repeats, _axis), do: nif()

  # --- Indexing ----------------------------------------------------

  @spec slice(tensor(), [integer()], [integer()], [integer()]) :: tensor()
  def slice(_a, _start, _stop, _strides), do: nif()

  @spec slice_update(tensor(), tensor(), [integer()]) :: tensor()
  def slice_update(_src, _update, _start), do: nif()

  @spec take(tensor(), tensor(), integer()) :: tensor()
  def take(_a, _indices, _axis), do: nif()

  @spec where(tensor(), tensor(), tensor()) :: tensor()
  def where(_cond, _x, _y), do: nif()

  # --- Linalg ------------------------------------------------------

  @spec matmul(tensor(), tensor()) :: tensor()
  def matmul(_a, _b), do: nif()

  @spec tensordot(tensor(), tensor(), [integer()], [integer()]) :: tensor()
  def tensordot(_a, _b, _axes_a, _axes_b), do: nif()

  @spec outer(tensor(), tensor()) :: tensor()
  def outer(_a, _b), do: nif()

  @spec inner(tensor(), tensor()) :: tensor()
  def inner(_a, _b), do: nif()

  # --- Sort --------------------------------------------------------

  @spec sort(tensor(), integer()) :: tensor()
  def sort(_a, _axis), do: nif()

  @spec argsort(tensor(), integer()) :: tensor()
  def argsort(_a, _axis), do: nif()

  @spec partition(tensor(), integer(), integer()) :: tensor()
  def partition(_a, _kth, _axis), do: nif()

  @spec argpartition(tensor(), integer(), integer()) :: tensor()
  def argpartition(_a, _kth, _axis), do: nif()

  @spec topk(tensor(), integer(), integer()) :: tensor()
  def topk(_a, _k, _axis), do: nif()

  # --- Misc --------------------------------------------------------

  @spec clip(tensor(), tensor(), tensor()) :: tensor()
  def clip(_a, _a_min, _a_max), do: nif()

  @spec roll(tensor(), integer(), integer()) :: tensor()
  def roll(_a, _shift, _axis), do: nif()

  @spec softmax(tensor(), [integer()], boolean()) :: tensor()
  def softmax(_a, _axes, _precise), do: nif()

  @spec logcumsumexp(tensor(), integer(), boolean(), boolean()) :: tensor()
  def logcumsumexp(_a, _axis, _reverse, _inclusive), do: nif()

  @spec array_equal(tensor(), tensor(), boolean()) :: tensor()
  def array_equal(_a, _b, _equal_nan), do: nif()

  # --- Axis-aligned gather/scatter ---------------------------------

  @spec take_along_axis(tensor(), tensor(), integer()) :: tensor()
  def take_along_axis(_a, _indices, _axis), do: nif()

  @spec put_along_axis(tensor(), tensor(), tensor(), integer()) :: tensor()
  def put_along_axis(_a, _indices, _values, _axis), do: nif()

  @spec scatter_add_axis(tensor(), tensor(), tensor(), integer()) :: tensor()
  def scatter_add_axis(_a, _indices, _values, _axis), do: nif()

  # --- Convolution -------------------------------------------------

  @spec conv_general(
          tensor(),
          tensor(),
          [integer()],
          {[integer()], [integer()]},
          [integer()],
          [integer()],
          integer(),
          boolean()
        ) :: tensor()
  def conv_general(
        _input,
        _weight,
        _stride,
        _padding,
        _kernel_dilation,
        _input_dilation,
        _groups,
        _flip
      ),
      do: nif()

  # --- Random ------------------------------------------------------

  @spec random_key(integer()) :: tensor()
  def random_key(_seed), do: nif()

  @spec random_split(tensor(), integer()) :: tensor()
  def random_split(_key, _num), do: nif()

  @spec random_uniform(tensor(), tensor(), [non_neg_integer()], dtype(), tensor() | nil) ::
          tensor()
  def random_uniform(_low, _high, _shape, _dtype, _key), do: nif()

  @spec random_normal([non_neg_integer()], dtype(), float(), float(), tensor() | nil) ::
          tensor()
  def random_normal(_shape, _dtype, _loc, _scale, _key), do: nif()

  @spec random_randint(tensor(), tensor(), [non_neg_integer()], dtype(), tensor() | nil) ::
          tensor()
  def random_randint(_low, _high, _shape, _dtype, _key), do: nif()

  @spec random_bernoulli(tensor(), [non_neg_integer()], tensor() | nil) :: tensor()
  def random_bernoulli(_p, _shape, _key), do: nif()

  @spec random_gumbel([non_neg_integer()], dtype(), tensor() | nil) :: tensor()
  def random_gumbel(_shape, _dtype, _key), do: nif()

  @spec random_categorical(tensor(), integer(), integer(), tensor() | nil) :: tensor()
  def random_categorical(_logits, _axis, _num_samples, _key), do: nif()

  # --- FFT ---------------------------------------------------------

  @spec fftn(tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def fftn(_a, _n, _axes), do: nif()

  @spec ifftn(tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def ifftn(_a, _n, _axes), do: nif()

  @spec rfftn(tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def rfftn(_a, _n, _axes), do: nif()

  @spec irfftn(tensor(), [non_neg_integer()], [integer()]) :: tensor()
  def irfftn(_a, _n, _axes), do: nif()

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
