defmodule Emily.Backend do
  @moduledoc """
  `Nx.Backend` implementation backed by Apple's MLX via `Emily.Native`.

  ## Usage

      Nx.global_default_backend(Emily.Backend)
      Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.sum()

  ## Deliberate divergences from Nx defaults

    * `{:f, 64}` is not supported — Metal cannot execute f64. Operations
      that would allocate an f64 tensor raise `ArgumentError` with a
      message pointing to f32.
    * `from_pointer`, `to_pointer`, `population_count`, and
      `count_leading_zeros` raise `ArgumentError` — MLX has no primitive.
    * Window operations (`window_sum`, `window_scatter_max`, etc.) and
      advanced linalg (`lu`, `svd`, `qr`, `cholesky`, `eigh`, `solve`,
      `determinant`, `triangular_solve`) fall back to Nx's default
      `optional/3` implementation — correct but slow.
    * `quotient` uses MLX `floor_divide` semantics (floor toward -inf
      rather than Nx's truncate-toward-zero). For non-negative integer
      operands the results agree; mixed-sign inputs diverge by one. We
      accept the divergence rather than paying for a software fixup on
      every int-division.
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref]
  defstruct [:ref]

  alias Emily.Backend, as: B
  alias Emily.Native
  alias Nx.Tensor, as: T

  @typep tensor :: T.t()
  @typep ref :: reference()

  # =================================================================
  # Helpers
  # =================================================================

  @spec ref(tensor()) :: ref()
  defp ref(%T{data: %B{ref: r}}), do: r

  # Tensor on a different backend (Nx routinely passes scalars on
  # BinaryBackend alongside our tensors). Transfer and recurse so the
  # ref extraction goes through the primary clause.
  defp ref(%T{} = t), do: t |> Nx.backend_transfer(Emily.Backend) |> ref()

  @spec wrap(ref(), tensor()) :: tensor()
  defp wrap(ref, %T{type: type} = out) do
    %{out | data: %B{ref: coerce(ref, type)}}
  end

  # Nx uses `{:u, 8}` to represent boolean outputs (comparisons,
  # logical ops, is_nan, …); MLX yields `mx::bool_` which Native
  # surfaces as `{:pred, 1}`. Same in-memory layout (1 byte/elem), but
  # Nx's dtype invariant fails downstream without the cast. MLX elides
  # a same-type astype, so the unconditional call is ~free when the
  # ref is already u8.
  defp coerce(ref, {:u, 8}), do: Native.astype(ref, {:u, 8})
  defp coerce(ref, _), do: ref

  defp shape_list(shape) when is_tuple(shape), do: Tuple.to_list(shape)

  defp check_dtype!({:f, 64}) do
    raise ArgumentError,
          "Emily.Backend does not support {:f, 64} — Metal has no f64. Use {:f, 32}."
  end

  defp check_dtype!(_type), do: :ok

  # Build a scalar Native tensor from an Elixir number (or :infinity /
  # :nan / :neg_infinity / Complex). We route through BinaryBackend to
  # get consistent encoding for f16/bf16/complex without duplicating
  # Nx's bit-packing logic.
  defp scalar_ref(value, type) do
    check_dtype!(type)
    scalar = Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
    bin = Nx.to_binary(scalar)
    Native.from_binary(bin, [], type)
  end

  # =================================================================
  # init
  # =================================================================

  @impl true
  def init(opts) do
    opts = Keyword.validate!(opts, device: :gpu)

    unless opts[:device] in [:cpu, :gpu] do
      raise ArgumentError,
            "Emily.Backend expected :device to be :cpu or :gpu, got: #{inspect(opts[:device])}"
    end

    opts
  end

  # =================================================================
  # Binary round-trip
  # =================================================================

  @impl true
  def from_binary(%T{shape: shape, type: type} = out, binary, _backend_options) do
    check_dtype!(type)
    bin = ensure_binary(binary)
    ref = Native.from_binary(bin, shape_list(shape), type)
    wrap(ref, out)
  end

  defp ensure_binary(b) when is_binary(b), do: b
  defp ensure_binary(iodata) when is_list(iodata), do: IO.iodata_to_binary(iodata)
  # Sub-byte bitstrings aren't iodata; fall through to list_to_bitstring.
  defp ensure_binary(bs) when is_bitstring(bs), do: :erlang.list_to_bitstring([bs])

  @impl true
  def to_binary(%T{data: %B{ref: r}, type: {_, bits}} = tensor, limit) do
    bin = Native.to_binary(r)
    elem_bits = effective_elem_bits(bits)
    size = Nx.size(tensor)

    if limit >= size do
      bin
    else
      binary_part(bin, 0, div(limit * elem_bits, 8))
    end
  end

  # Nx counts pred as 1 bit, but MLX stores bool_ as 1 byte. At the
  # binary layer we always see bytes.
  defp effective_elem_bits(1), do: 8
  defp effective_elem_bits(bits), do: bits

  # =================================================================
  # Backend ownership
  # =================================================================

  @impl true
  def backend_deallocate(_tensor), do: :ok

  @impl true
  def backend_copy(tensor, backend, opts), do: backend_transfer(tensor, backend, opts)

  @impl true
  def backend_transfer(tensor, Nx.Tensor, _opts), do: tensor
  def backend_transfer(tensor, Emily.Backend, _opts), do: tensor

  def backend_transfer(%T{} = tensor, backend, opts) do
    binary = to_binary(tensor, Nx.size(tensor))
    backend.from_binary(tensor, binary, opts)
  end

  @impl true
  def from_pointer(_pointer, _type, _shape, _backend_opts, _opts),
    do:
      raise(
        ArgumentError,
        "Emily.Backend does not implement pointer manipulation (no safe MLX equivalent)"
      )

  @impl true
  def to_pointer(_tensor, _opts),
    do:
      raise(
        ArgumentError,
        "Emily.Backend does not implement pointer manipulation (no safe MLX equivalent)"
      )

  # =================================================================
  # Inspect
  # =================================================================

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = inspect_opts.limit

    binary =
      case limit do
        :infinity -> Nx.to_binary(tensor)
        n -> Nx.to_binary(tensor, limit: min(n + 1, Nx.size(tensor)))
      end

    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  # =================================================================
  # to_batched
  # =================================================================

  @impl true
  def to_batched(%T{shape: out_shape} = out, %T{shape: in_shape} = tensor, opts) do
    leftover = opts[:leftover] || :discard
    batch_size = elem(out_shape, 0)
    axis_size = elem(in_shape, 0)

    num_full = div(axis_size, batch_size)
    remainder = rem(axis_size, batch_size)

    range =
      if remainder != 0 and leftover == :repeat do
        0..num_full
      else
        0..(num_full - 1)
      end

    binary = to_binary(tensor, Nx.size(tensor))
    {_, type_bits} = tensor.type
    elem_bits = effective_elem_bits(type_bits)
    batch_bytes = div(Nx.size(out) * elem_bits, 8)

    Stream.map(range, fn
      ^num_full ->
        before = num_full * batch_bytes
        available = byte_size(binary) - before
        missing = batch_bytes - available
        wrapped = binary_part(binary, before, available) <> binary_part(binary, 0, missing)
        from_binary(out, wrapped, [])

      i ->
        slice = binary_part(binary, i * batch_bytes, batch_bytes)
        from_binary(out, slice, [])
    end)
  end

  # =================================================================
  # Creation
  # =================================================================

  @impl true
  def constant(%T{shape: {}, type: type} = out, value, _opts) do
    check_dtype!(type)
    scalar_ref(value, type) |> wrap(out)
  end

  def constant(%T{shape: shape, type: type} = out, value, _opts) do
    check_dtype!(type)
    scalar = scalar_ref(value, type)
    Native.full(shape_list(shape), scalar, type) |> wrap(out)
  end

  @impl true
  def iota(%T{shape: {}, type: type} = out, _axis, _opts) do
    check_dtype!(type)
    scalar_ref(0, type) |> wrap(out)
  end

  def iota(%T{shape: shape, type: type} = out, nil, _opts) do
    check_dtype!(type)
    size = Nx.size(shape)

    Native.arange(0.0, size * 1.0, 1.0, type)
    |> Native.reshape(shape_list(shape))
    |> wrap(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis, _opts) do
    check_dtype!(type)
    dims = Tuple.to_list(shape)
    dim = Enum.at(dims, axis)

    # Build a 1-D arange along the chosen axis, reshape to a thin
    # (1, 1, ..., dim, ..., 1, 1) and broadcast to the full shape.
    line = Native.arange(0.0, dim * 1.0, 1.0, type)

    thin_shape =
      dims
      |> Enum.with_index()
      |> Enum.map(fn {_, i} -> if i == axis, do: dim, else: 1 end)

    line
    |> Native.reshape(thin_shape)
    |> Native.broadcast_to(dims)
    |> wrap(out)
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, _opts) do
    check_dtype!(type)
    rank = tuple_size(shape)
    n = elem(shape, rank - 2)
    m = elem(shape, rank - 1)
    base = Native.eye(n, m, 0, type)

    if rank == 2 do
      base |> wrap(out)
    else
      base
      |> Native.broadcast_to(shape_list(shape))
      |> wrap(out)
    end
  end

  # =================================================================
  # Cast
  # =================================================================

  @impl true
  def as_type(%T{type: type} = out, %T{} = t) do
    check_dtype!(type)
    t |> ref() |> Native.astype(type) |> wrap(out)
  end

  # bitcast: reinterpret the bits as the output dtype (same element
  # size). MLX exposes this as `mx::view`. Used by Nx.Random to move
  # between uint and float of matching width.
  @impl true
  def bitcast(%T{type: type} = out, t) do
    t |> ref() |> Native.bitcast(type) |> wrap(out)
  end

  # =================================================================
  # Unary ops
  # =================================================================

  # Nx callback name → Native NIF name, for ops where the two disagree.
  # Same-named ops are declared below via @direct_unary.
  @renamed_unary [
    negate: :negative,
    bitwise_not: :bitwise_invert,
    is_nan: :isnan,
    is_infinity: :isinf,
    acos: :arccos,
    asin: :arcsin,
    atan: :arctan,
    acosh: :arccosh,
    asinh: :arcsinh,
    atanh: :arctanh,
    erf_inv: :erfinv
  ]

  for {nx_name, native_name} <- @renamed_unary do
    @impl true
    def unquote(nx_name)(out, t) do
      t |> ref() |> Native.unquote(native_name)() |> wrap(out)
    end
  end

  @direct_unary ~w(abs ceil floor sign sqrt rsqrt exp expm1 log log1p
                   sin cos tan sinh cosh tanh sigmoid erf conjugate
                   real imag)a

  for op <- @direct_unary do
    @impl true
    def unquote(op)(out, t) do
      t |> ref() |> Native.unquote(op)() |> wrap(out)
    end
  end

  @impl true
  def round(out, t), do: t |> ref() |> Native.round(0) |> wrap(out)

  # erfc(x) = 1 - erf(x) — composed in Elixir.
  @impl true
  def erfc(%T{type: type} = out, t) do
    r = ref(t)
    one = scalar_ref(1, type)
    erf_r = Native.erf(r)
    Native.subtract(one, erf_r) |> wrap(out)
  end

  # cbrt(x) = sign(x) * |x|^(1/3). MLX has no cbrt primitive.
  @impl true
  def cbrt(%T{type: type} = out, t) do
    r = ref(t)
    signed = Native.sign(r)
    absolute = Native.abs(r)
    third = scalar_ref(1.0 / 3.0, type)
    pow = Native.power(absolute, third)
    Native.multiply(signed, pow) |> wrap(out)
  end

  @impl true
  def count_leading_zeros(_out, _t),
    do:
      raise(
        ArgumentError,
        "Emily.Backend does not implement count_leading_zeros (MLX has no primitive)"
      )

  @impl true
  def population_count(_out, _t),
    do:
      raise(
        ArgumentError,
        "Emily.Backend does not implement population_count (MLX has no primitive)"
      )

  # logical_not is listed under @optional_callbacks in Nx.Backend; we
  # implement it directly.
  @impl true
  def logical_not(out, t), do: t |> ref() |> Native.logical_not() |> wrap(out)

  # =================================================================
  # Binary ops
  # =================================================================

  # Arithmetic + bitwise: cast both operands to `out.type` before
  # handing to MLX. Two reasons:
  #   1. MLX's cross-type promotion for mixed integer widths (e.g.,
  #      u64 + s32) falls back to float32 — which then fails on
  #      integer-only ops like right_shift. `Nx.Random.key` hits this.
  #   2. `divide` has `out.type = float` even for integer operands
  #      (`Nx.Type.to_floating/1`); casting to out.type first produces
  #      the float division Nx promises.
  @renamed_arith_binary [
    subtract: :subtract,
    multiply: :multiply,
    divide: :divide,
    remainder: :remainder,
    pow: :power,
    atan2: :arctan2,
    min: :minimum,
    max: :maximum,
    bitwise_and: :bitwise_and,
    bitwise_or: :bitwise_or,
    bitwise_xor: :bitwise_xor,
    left_shift: :left_shift,
    right_shift: :right_shift
  ]

  for {nx_name, native_name} <- @renamed_arith_binary do
    @impl true
    def unquote(nx_name)(%T{type: type} = out, a, b) do
      ra = Native.astype(ref(a), type)
      rb = Native.astype(ref(b), type)
      Native.unquote(native_name)(ra, rb) |> wrap(out)
    end
  end

  # Compare + logical: out.type is `{:u, 8}` (pred), but MLX still
  # needs the operands at a matched non-pred type to compare. Cast to
  # `Nx.Type.merge(a, b)` so MLX sees a consistent arithmetic type.
  @renamed_pred_binary [
    equal: :equal,
    not_equal: :not_equal,
    less: :less,
    less_equal: :less_equal,
    greater: :greater,
    greater_equal: :greater_equal,
    logical_and: :logical_and,
    logical_or: :logical_or
  ]

  for {nx_name, native_name} <- @renamed_pred_binary do
    @impl true
    def unquote(nx_name)(out, a, b) do
      target = Nx.Type.merge(a.type, b.type)
      ra = Native.astype(ref(a), target)
      rb = Native.astype(ref(b), target)
      Native.unquote(native_name)(ra, rb) |> wrap(out)
    end
  end

  @impl true
  def add(%T{type: type} = out, a, b) do
    ra = Native.astype(ref(a), type)
    rb = Native.astype(ref(b), type)
    Native.add(ra, rb) |> wrap(out)
  end

  # See moduledoc: we intentionally use MLX floor_divide for quotient,
  # matching its rounding (floor toward -inf) rather than Nx's
  # truncate-toward-zero. The two agree for non-negative operands.
  @impl true
  def quotient(%T{type: type} = out, a, b) do
    ra = Native.astype(ref(a), type)
    rb = Native.astype(ref(b), type)
    Native.floor_divide(ra, rb) |> wrap(out)
  end

  # logical_xor: MLX has no direct op. Compose via `not_equal` on
  # booleanised inputs. With u8 inputs, `not_equal(0)` yields truthy
  # masks; their `not_equal` is the xor.
  @impl true
  def logical_xor(out, a, b) do
    ra = ref(a)
    rb = ref(b)
    za = scalar_ref(0, a.type)
    zb = scalar_ref(0, b.type)
    ma = Native.not_equal(ra, za)
    mb = Native.not_equal(rb, zb)
    Native.not_equal(ma, mb) |> wrap(out)
  end

  # =================================================================
  # Shape
  # =================================================================

  @impl true
  def reshape(%T{shape: shape} = out, t),
    do: t |> ref() |> Native.reshape(shape_list(shape)) |> wrap(out)

  @impl true
  def squeeze(out, t, axes), do: t |> ref() |> Native.squeeze(axes) |> wrap(out)

  @impl true
  def transpose(out, t, axes), do: t |> ref() |> Native.transpose(axes) |> wrap(out)

  # Nx broadcast: given input tensor shape `in_shape` and output `shape`,
  # `axes` is the positions in `shape` where `in_shape`'s axes land.
  # MLX's broadcast_to wants the input already reshaped to align with
  # the output. We reshape first (inserting singletons at the missing
  # positions), then broadcast_to.
  @impl true
  def broadcast(%T{shape: out_shape} = out, t, _shape, axes) do
    in_shape = t.shape
    in_dims = Tuple.to_list(in_shape)
    out_dims = Tuple.to_list(out_shape)

    # Build an intermediate shape of rank == length(out_dims) with 1s
    # everywhere except the axes in `axes`, where we place the
    # corresponding dim from `in_shape`.
    placed = Enum.zip(axes, in_dims) |> Map.new()

    intermediate =
      out_dims
      |> Enum.with_index()
      |> Enum.map(fn {_, i} -> Map.get(placed, i, 1) end)

    t
    |> ref()
    |> Native.reshape(intermediate)
    |> Native.broadcast_to(out_dims)
    |> wrap(out)
  end

  # Nx padding_config: [{low, high, interior}, ...]. MLX supports low/high
  # but not interior dilation; reject > 0.
  @impl true
  def pad(%T{} = out, t, %T{} = pad_value, padding_config) do
    lows = Enum.map(padding_config, fn {lo, _, _} -> lo end)
    highs = Enum.map(padding_config, fn {_, hi, _} -> hi end)
    interiors = Enum.map(padding_config, fn {_, _, interior} -> interior end)

    if Enum.any?(interiors, &(&1 > 0)) do
      raise ArgumentError,
            "Emily.Backend does not implement interior padding (MLX has no primitive)"
    end

    axes = Enum.to_list(0..(length(lows) - 1))
    Native.pad(ref(t), axes, lows, highs, ref(pad_value)) |> wrap(out)
  end

  # Reverse along each axis in `axes`. MLX's C++ slice doesn't interpret
  # negative stops the way numpy does, so we build a reverse-index
  # tensor per axis and `take`.
  @impl true
  def reverse(%T{} = out, t, axes) do
    reversed =
      Enum.reduce(axes, ref(t), fn axis, acc ->
        dim = elem(t.shape, axis)
        indices = reverse_indices(dim)
        Native.take(acc, indices, axis)
      end)

    wrap(reversed, out)
  end

  defp reverse_indices(dim) do
    Native.arange(dim * 1.0 - 1.0, -1.0, -1.0, {:s, 32})
  end

  @impl true
  def concatenate(%T{} = out, tensors, axis) do
    refs = Enum.map(tensors, &ref/1)
    Native.concatenate(refs, axis) |> wrap(out)
  end

  @impl true
  def stack(%T{} = out, tensors, axis) do
    refs = Enum.map(tensors, &ref/1)
    Native.stack(refs, axis) |> wrap(out)
  end

  # =================================================================
  # Indexing
  # =================================================================

  @impl true
  def slice(%T{} = out, t, starts, lengths, strides) do
    # Nx passes starts as either integers or scalar tensors (dynamic
    # slicing). MLX's slice takes integer bounds; under the evaluator
    # we materialise scalar-tensor starts to their concrete value.
    starts = Enum.map(starts, &slice_start/1)
    stops = Enum.zip_with(starts, lengths, fn s, l -> s + l end)
    Native.slice(ref(t), starts, stops, strides) |> wrap(out)
  end

  defp slice_start(i) when is_integer(i), do: i
  defp slice_start(%T{} = t), do: t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_number()

  # put_slice: implemented natively via MLX `slice_update`. Nx promotes
  # operand types at the API layer — `Nx.put_slice(s32_buf, _, s64_upd)`
  # declares an s64 output — but our callback arguments still carry the
  # original backend types. We cast both `t` and `slice` to `out.type`
  # before dispatching so the MLX buffer matches Nx's shape/type view.
  # Scalar-tensor starts are materialised to integers here (dynamic
  # indices show up when autoregressive loops dispatch put_slice from
  # within `defn`).
  @impl true
  def put_slice(%T{type: type} = out, %T{} = t, start_indices, %T{} = slice) do
    starts = Enum.map(start_indices, &slice_start/1)
    src_ref = Native.astype(ref(t), type)
    update_ref = Native.astype(ref(slice), type)
    Native.slice_update(src_ref, update_ref, starts) |> wrap(out)
  end

  @impl true
  def select(%T{} = out, pred, on_true, on_false) do
    # MLX where expects a bool condition.
    cond_ref = Native.astype(ref(pred), {:pred, 1})
    Native.where(cond_ref, ref(on_true), ref(on_false)) |> wrap(out)
  end

  @impl true
  def clip(%T{} = out, t, min_t, max_t) do
    Native.clip(ref(t), ref(min_t), ref(max_t)) |> wrap(out)
  end

  # gather: Nx's gather takes a multi-dimensional index tensor whose
  # last dim selects across multiple axes. MLX's take_along_axis works
  # on a single axis. The single-axis case (embedding lookups) uses
  # Native.take; the general case falls back to BinaryBackend.
  @impl true
  def gather(out, input, indices, opts) do
    case opts[:axes] do
      [axis] ->
        idx_ref = ref(indices) |> Native.astype({:s, 32})
        Native.take(ref(input), idx_ref, axis) |> wrap(out)

      _ ->
        via_binary(out, [input, indices], &Nx.gather(&1, &2, opts))
    end
  end

  # =================================================================
  # Reductions
  # =================================================================

  for {nx_name, native_name} <- [
        sum: :sum,
        product: :prod,
        all: :all,
        any: :any,
        reduce_max: :max,
        reduce_min: :min
      ] do
    @impl true
    def unquote(nx_name)(out, t, opts) do
      axes = reduction_axes(opts, t)
      keep = opts[:keep_axes] || false
      Native.unquote(native_name)(ref(t), axes, keep) |> wrap(out)
    end
  end

  # Nx reductions accept `:axes` (list) and `:keep_axes` (bool). When
  # `:axes` is nil, reduce across all axes.
  defp reduction_axes(opts, %T{shape: shape}) do
    case opts[:axes] do
      nil -> Enum.to_list(0..(tuple_size(shape) - 1))
      list -> list
    end
  end

  # Nx's argmax/argmin take `:keep_axis` (singular) on user-facing API
  # but the backend callback exposes raw opts whose spelling has drifted
  # across Nx versions. Derive `keep` from the shape invariant instead:
  # if `out.shape` has the same rank as the input, the axis was kept.
  @impl true
  def argmax(%T{} = out, t, opts) do
    axis = opts[:axis] || 0
    keep = tuple_size(out.shape) == tuple_size(t.shape)

    ref(t)
    |> Native.argmax(axis, keep)
    |> Native.astype(out.type)
    |> wrap(out)
  end

  @impl true
  def argmin(%T{} = out, t, opts) do
    axis = opts[:axis] || 0
    keep = tuple_size(out.shape) == tuple_size(t.shape)

    ref(t)
    |> Native.argmin(axis, keep)
    |> Native.astype(out.type)
    |> wrap(out)
  end

  # Cumulative reductions are optional callbacks in Nx. We implement
  # them directly via `Native` for the fast path (cumulate along the
  # last axis), and fall back to BinaryBackend for interior-axis
  # cumulation.
  #
  # MLX's cumulative kernels raise "Unable to safely factor shape" on
  # some view patterns — notably interior-axis cumulation on 4-D+
  # tensors. Transposing the target axis to the end first usually
  # works but hits the same factoring issue on a subset of shapes, so
  # we route those through BinaryBackend: correct, slow, rare.
  for {nx_name, native_name} <- [
        cumulative_sum: :cumsum,
        cumulative_product: :cumprod,
        cumulative_max: :cummax,
        cumulative_min: :cummin
      ] do
    @impl true
    def unquote(nx_name)(%T{} = out, t, opts) do
      axis = opts[:axis] || 0
      reverse = opts[:reverse] || false
      rank = tuple_size(t.shape)

      if axis == rank - 1 do
        Native.unquote(native_name)(ref(t), axis, reverse, true) |> wrap(out)
      else
        nx_fun = unquote(nx_name)
        via_binary(out, [t], &apply(Nx, nx_fun, [&1, opts]))
      end
    end
  end

  # =================================================================
  # Dot product
  # =================================================================

  # Non-batched: tensordot. Batched: permute to [batch, free, contract]
  # on a and [batch, contract, free] on b, flatten to 3-D, hand to
  # MLX matmul (which treats leading dims as batch), reshape back to
  # Nx's canonical `batch ++ free_a ++ free_b` layout.
  @impl true
  def dot(%T{} = out, a, contract_a, [], b, contract_b, []) do
    Native.tensordot(ref(a), ref(b), contract_a, contract_b) |> wrap(out)
  end

  def dot(%T{type: type} = out, a, contract_a, batch_a, b, contract_b, batch_b) do
    # MLX matmul is float-only; ints/preds fall through to BinaryBackend.
    # In practice every transformer-attention call is float, so this is
    # the hot path.
    if float_like?(type) do
      batched_matmul(out, a, contract_a, batch_a, b, contract_b, batch_b)
    else
      via_binary(out, [a, b], &Nx.dot(&1, contract_a, batch_a, &2, contract_b, batch_b))
    end
  end

  defp float_like?({kind, _}) when kind in [:f, :bf, :c], do: true
  defp float_like?(_), do: false

  # Nx guarantees batch axes on both tensors are [0, 1, ..., k-1] in
  # increasing order, so the permutation simplifies: batch dims stay
  # at the front, free axes sort in positional order, contract axes
  # in the Nx-given pairing order.
  defp batched_matmul(
         %T{shape: out_shape} = out,
         %T{shape: as} = a,
         contract_a,
         batch_a,
         %T{shape: bs} = b,
         contract_b,
         _batch_b
       ) do
    a_rank = tuple_size(as)
    b_rank = tuple_size(bs)
    k = length(batch_a)

    contract_set_a = MapSet.new(contract_a)
    contract_set_b = MapSet.new(contract_b)

    free_a = for i <- k..(a_rank - 1)//1, not MapSet.member?(contract_set_a, i), do: i
    free_b = for i <- k..(b_rank - 1)//1, not MapSet.member?(contract_set_b, i), do: i

    b_prod = dim_product(batch_a, as)
    m = dim_product(free_a, as)
    n = dim_product(free_b, bs)
    k_prod = dim_product(contract_a, as)

    perm_a = batch_a ++ free_a ++ contract_a
    perm_b = batch_a ++ contract_b ++ free_b

    ra =
      a
      |> ref()
      |> Native.transpose(perm_a)
      |> Native.reshape([b_prod, m, k_prod])

    rb =
      b
      |> ref()
      |> Native.transpose(perm_b)
      |> Native.reshape([b_prod, k_prod, n])

    Native.matmul(ra, rb)
    |> Native.reshape(shape_list(out_shape))
    |> wrap(out)
  end

  defp dim_product(axes, shape), do: Enum.reduce(axes, 1, &(elem(shape, &1) * &2))

  # =================================================================
  # Sort / argsort / top_k / all_close / take / take_along_axis
  # =================================================================

  @impl true
  def sort(%T{} = out, t, opts) do
    axis = opts[:axis] || 0
    direction = opts[:direction] || :asc

    sorted = Native.sort(ref(t), axis)

    case direction do
      :asc -> sorted |> wrap(out)
      :desc -> flip_axis(sorted, t.shape, axis) |> wrap(out)
    end
  end

  @impl true
  def argsort(%T{} = out, t, opts) do
    axis = opts[:axis] || 0
    direction = opts[:direction] || :asc

    idx = Native.argsort(ref(t), axis)

    idx =
      case direction do
        :asc -> idx
        :desc -> flip_axis(idx, t.shape, axis)
      end

    idx |> Native.astype(out.type) |> wrap(out)
  end

  # Flip a single axis by take-with-reversed-indices. Same reasoning
  # as `reverse/3`.
  defp flip_axis(ref, shape, axis) do
    dim = elem(shape, axis)
    Native.take(ref, reverse_indices(dim), axis)
  end

  # MLX topk returns k largest values along the last axis, unsorted;
  # Nx expects them sorted descending.
  @impl true
  def top_k(%T{} = out, t, opts) do
    k = opts[:k]
    axis = tuple_size(t.shape) - 1

    ref(t)
    |> Native.topk(k, -1)
    |> Native.sort(axis)
    |> flip_axis(out.shape, axis)
    |> wrap(out)
  end

  # all_close with absolute/relative tolerance:
  # all(abs(a - b) <= atol + rtol * abs(b)).
  @impl true
  def all_close(%T{} = out, a, b, opts) do
    rtol = opts[:rtol] || 1.0e-5
    atol = opts[:atol] || 1.0e-8
    equal_nan = opts[:equal_nan] || false

    t = Nx.Type.merge(a.type, b.type) |> Nx.Type.to_floating()
    check_dtype!(t)

    ra = Native.astype(ref(a), t)
    rb = Native.astype(ref(b), t)

    diff = Native.abs(Native.subtract(ra, rb))
    tol = Native.add(scalar_ref(atol, t), Native.multiply(scalar_ref(rtol, t), Native.abs(rb)))
    close = Native.less_equal(diff, tol)

    close =
      if equal_nan do
        both_nan = Native.logical_and(Native.isnan(ra), Native.isnan(rb))
        Native.logical_or(close, both_nan)
      else
        close
      end

    axes = Enum.to_list(0..(tuple_size(a.shape) - 1))
    Native.all(close, axes, false) |> wrap(out)
  end

  @impl true
  def take(%T{} = out, input, indices, opts) do
    axis = opts[:axis] || 0
    idx_ref = ref(indices) |> Native.astype({:s, 32})
    Native.take(ref(input), idx_ref, axis) |> wrap(out)
  end

  @impl true
  def take_along_axis(%T{} = out, input, indices, opts) do
    axis = opts[:axis] || 0
    idx_ref = ref(indices) |> Native.astype({:s, 32})
    Native.take_along_axis(ref(input), idx_ref, axis) |> wrap(out)
  end

  # =================================================================
  # FFT
  # =================================================================

  @impl true
  def fft(%T{} = out, t, opts) do
    length = opts[:length]
    axis = tuple_size(t.shape) - 1
    Native.fftn(ref(t), [length], [axis]) |> wrap(out)
  end

  @impl true
  def ifft(%T{} = out, t, opts) do
    length = opts[:length]
    axis = tuple_size(t.shape) - 1
    Native.ifftn(ref(t), [length], [axis]) |> wrap(out)
  end

  @impl true
  def fft2(%T{} = out, t, opts) do
    lengths = opts[:lengths]
    axes = opts[:axes]
    Native.fftn(ref(t), lengths, axes) |> wrap(out)
  end

  @impl true
  def ifft2(%T{} = out, t, opts) do
    lengths = opts[:lengths]
    axes = opts[:axes]
    Native.ifftn(ref(t), lengths, axes) |> wrap(out)
  end

  # =================================================================
  # Conv (delegated to BinaryBackend for M2)
  # =================================================================
  #
  # Nx's conv signature is rich (batch groups, feature groups,
  # permutations, dilations). MLX conv_general covers most of this but
  # the translation is involved. For M2 we go through BinaryBackend;
  # native conv lands in M3 alongside Bumblebee integration.

  # Conv lands natively via Native.conv_general in M3 alongside
  # Bumblebee integration. Nx.conv on BinaryBackend is correct but
  # CPU-bound — fine for tests, unusable for real workloads.
  @impl true
  def conv(out, input, kernel, opts),
    do: via_binary(out, [input, kernel], &Nx.conv(&1, &2, opts))

  # =================================================================
  # Unsupported / fallback callbacks
  # =================================================================
  #
  # These either have no MLX primitive or a general implementation that
  # would be substantial work. Routed through BinaryBackend: transfer
  # inputs, run the reference op, transfer the result back. Correct
  # but slow; M3+ replaces the performance-critical ones (batched
  # `dot`, `conv`) with direct MLX calls.

  # Run `fun` on BinaryBackend-transferred copies of `tensors` and wrap
  # the single-tensor result into `out`.
  #
  # We pin the default backend to `Nx.BinaryBackend` for the duration
  # of `fun` because some Nx ops build scalar tensors internally
  # (e.g. `Nx.conv` constructs a zero-pad tensor via `Nx.pad(t, 0,
  # ...)`; `Nx.indexed_add` wraps the accumulator). Without the pin,
  # those scalars land on the current global default — which is
  # `Emily.Backend` during conformance tests — and the resulting
  # mixed-backend operand list crashes inside BinaryBackend's op.
  defp via_binary(%T{} = out, tensors, fun) when is_list(tensors) do
    result =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        tensors |> transfer_all() |> then(&apply(fun, &1))
      end)

    from_binary(out, Nx.to_binary(result), [])
  end

  # Same pattern, but the op returns a tuple of tensors. `outs` is a
  # tuple of output templates matching arity; positions are zipped.
  defp via_binary_tuple(outs, tensors, fun) when is_tuple(outs) and is_list(tensors) do
    result_tuple =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        tensors |> transfer_all() |> then(&apply(fun, &1))
      end)

    outs
    |> Tuple.to_list()
    |> Enum.zip(Tuple.to_list(result_tuple))
    |> Enum.map(fn {out, r} -> from_binary(out, Nx.to_binary(r), []) end)
    |> List.to_tuple()
  end

  defp transfer_all(tensors),
    do: Enum.map(tensors, &Nx.backend_transfer(&1, Nx.BinaryBackend))

  @impl true
  def reduce(out, t, acc, opts, fun),
    do: via_binary(out, [t, acc], &Nx.reduce(&1, &2, opts, fun))

  @impl true
  def window_reduce(out, t, acc, window_shape, opts, fun),
    do: via_binary(out, [t, acc], &Nx.window_reduce(&1, &2, window_shape, opts, fun))

  for {nx_name, nx_fun} <- [
        window_sum: :window_sum,
        window_product: :window_product,
        window_max: :window_max,
        window_min: :window_min
      ] do
    @impl true
    def unquote(nx_name)(out, t, window_shape, opts) do
      via_binary(out, [t], &apply(Nx, unquote(nx_fun), [&1, window_shape, opts]))
    end
  end

  for {nx_name, nx_fun} <- [
        window_scatter_max: :window_scatter_max,
        window_scatter_min: :window_scatter_min
      ] do
    @impl true
    def unquote(nx_name)(out, t, source, init, window_shape, opts) do
      via_binary(
        out,
        [t, source, init],
        &apply(Nx, unquote(nx_fun), [&1, &2, &3, window_shape, opts])
      )
    end
  end

  @impl true
  def indexed_add(out, t, indices, updates, opts),
    do: via_binary(out, [t, indices, updates], &Nx.indexed_add(&1, &2, &3, opts))

  @impl true
  def indexed_put(out, t, indices, updates, opts),
    do: via_binary(out, [t, indices, updates], &Nx.indexed_put(&1, &2, &3, opts))

  @impl true
  def lu(outs, t, opts), do: via_binary_tuple(outs, [t], &Nx.LinAlg.lu(&1, opts))

  @impl true
  def triangular_solve(out, a, b, opts),
    do: via_binary(out, [a, b], &Nx.LinAlg.triangular_solve(&1, &2, opts))

  @impl true
  def svd(outs, t, opts), do: via_binary_tuple(outs, [t], &Nx.LinAlg.svd(&1, opts))
end
