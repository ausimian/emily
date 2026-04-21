defmodule Emily.Backend do
  @moduledoc """
  `Nx.Backend` implementation backed by Apple's MLX.

  ## Public API

  Users rarely call functions on this module directly. Install it as
  the default backend (or the per-tensor `backend:` opt) and `Nx` does
  the dispatch:

      Nx.global_default_backend({Emily.Backend, device: :gpu})

      Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      |> Nx.sum()
      |> Nx.to_flat_list()
      # => [10.0]

  Every function defined here implements a callback from the
  `Nx.Backend` behaviour (see `@impl true` in the source); they form
  the Nx dispatch table, not a user-facing API. The handful of
  `fast_*` functions are the dispatch targets for optional-expression
  nodes emitted by `Emily.Fast` — again internal.

  ## Options

    * `:device` — `:gpu` (default) or `:cpu`. Stored per-tensor; MLX
      dispatches the computation on that device.

  ## Divergences from `Nx.BinaryBackend`

    * `{:f, 64}` is not supported — Metal cannot execute f64.
      Allocations at f64 raise `ArgumentError`; cast to f32 instead.
    * `from_pointer`, `to_pointer`, `population_count`,
      `count_leading_zeros`, and interior-padding `pad` raise —
      MLX has no primitive.
    * `qr` with `mode: :complete` falls back to `Nx.BinaryBackend`
      (MLX only supports reduced QR). `determinant` uses Nx's default
      implementation, which calls the native `lu` for matrices
      larger than 3×3.
    * `quotient` uses MLX `floor_divide` semantics (floor toward -∞
      rather than Nx's truncate-toward-zero). For non-negative integer
      operands the results agree; mixed-sign inputs diverge by one.
    * Duplicate-index `indexed_put`: MLX's underlying scatter is
      unordered on duplicates, while `Nx.BinaryBackend` is
      deterministic last-write. `indexed_add` is commutative so
      duplicates accumulate identically on both backends.

  ## Fallbacks

  A handful of ops have no direct MLX primitive and fall back to
  `Nx.BinaryBackend` via a transparent round-trip
  (`from_pointer`-free, one memcpy each way). The fallback emits
  `[:emily, :fallback, *]` telemetry spans; see `Emily.Telemetry` for
  the full catalogue and for opt-in one-shot warnings.

  Every op below has a native MLX path for its hot shape/dtype and
  falls back only when the input hits the listed guard:

    * `gather` — indices tensor not in the
      `{batch…, rank_of_axes}` layout MLX gather accepts.
    * `cumulative_sum` / `cumulative_product` / `cumulative_max` /
      `cumulative_min` — when `axis` is not the last axis (MLX's
      factoring raises on some interior-axis views).
    * `dot` — batched dot on integer / pred types (MLX matmul is
      float-only). The non-batched tensordot path handles ints
      natively.
    * `conv` — `batch_group_size > 1`, or complex-typed.
    * `reduce` — always, since the reducer is a user-supplied BEAM
      function that can't be JITed into Metal.
    * `window_reduce` — same reason. The fixed `window_sum` /
      `window_product` / `window_max` / `window_min` variants all run
      native.
    * `indexed_add` / `indexed_put` — indices tensor not in MLX's
      native scatter layout.
    * `qr` with `mode: :complete`. `mode: :reduced` is native.

  ## Debug assertions

  Compile-time flags `:debug_bounds_check` and `:debug_detect_nan_inf`
  re-enable runtime assertions on hot paths. Both default to `false`
  with zero cost. See `Emily` moduledoc for details.
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref]
  defstruct [:ref]

  alias Emily.Backend, as: B
  alias Emily.Backend.DebugHelpers
  alias Emily.Native
  alias Nx.Tensor, as: T

  # Compile-time debug flags (M22). Default `false` so the `if @flag`
  # gates below fold at compile time — the `DebugHelpers.*` references
  # never appear in this module's BEAM when the flags are off. See
  # `Emily`'s moduledoc and `test/emily/debug_flags_test.exs`.
  @debug_bounds_check Application.compile_env(:emily, :debug_bounds_check, false)
  @debug_detect_nan_inf Application.compile_env(:emily, :debug_detect_nan_inf, false)

  @typep tensor :: T.t()
  @typep ref :: reference()

  # The four callbacks below are genuinely unimplementable on MLX and
  # exist solely to raise — dialyzer's `:error_handling` flag would
  # otherwise flag every one as "only terminates with explicit exception".
  @dialyzer {:nowarn_function,
             [
               from_pointer: 5,
               to_pointer: 2,
               count_leading_zeros: 2,
               population_count: 2
             ]}

  # =================================================================
  # Helpers
  # =================================================================

  @spec ref(tensor()) :: ref()
  defp ref(%T{data: %B{ref: r}}), do: r

  # Tensor on a different backend (Nx routinely passes scalars on
  # BinaryBackend alongside our tensors). Transfer and recurse so the
  # ref extraction goes through the primary clause.
  defp ref(%T{} = t), do: t |> Nx.backend_transfer(Emily.Backend) |> ref()

  @spec wrap(ref(), tensor(), reference()) :: tensor()
  defp wrap(ref, %T{} = out, w) do
    %{out | data: %B{ref: coerce(ref, out.type, w)}}
  end

  # Fast path: pred→u8 is the most common mismatch (MLX comparison/logical
  # ops yield bool; Nx expects u8). The general path covers bf16→f32 type
  # promotion from Nx.Defn.grad and any other MLX/Nx dtype disagreement.
  defp coerce(ref, {:u, 8}, w), do: Native.astype(w, ref, {:u, 8})

  defp coerce(ref, type, w) do
    case Native.dtype(ref) do
      ^type -> ref
      _ -> Native.astype(w, ref, type)
    end
  end

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

  defp worker, do: Emily.MlxStream.default_worker()

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
    wrap(ref, out, worker())
  end

  defp ensure_binary(b) when is_binary(b), do: b
  defp ensure_binary(iodata) when is_list(iodata), do: IO.iodata_to_binary(iodata)
  # Sub-byte bitstrings aren't iodata; fall through to list_to_bitstring.
  defp ensure_binary(bs) when is_bitstring(bs), do: :erlang.list_to_bitstring([bs])

  @impl true
  def to_binary(%T{data: %B{ref: r}, shape: shape, type: type} = tensor, limit) do
    metadata = %{shape: shape, dtype: type}

    bin =
      :telemetry.span([:emily, :to_binary], metadata, fn ->
        bytes = Native.to_binary(worker(), r)
        {bytes, Map.put(metadata, :byte_size, byte_size(bytes))}
      end)

    {_, bits} = type
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
    scalar_ref(value, type) |> wrap(out, worker())
  end

  def constant(%T{shape: shape, type: type} = out, value, _opts) do
    check_dtype!(type)
    w = worker()
    scalar = scalar_ref(value, type)
    Native.full(w, shape_list(shape), scalar, type) |> wrap(out, w)
  end

  @impl true
  def iota(%T{shape: {}, type: type} = out, _axis, _opts) do
    check_dtype!(type)
    scalar_ref(0, type) |> wrap(out, worker())
  end

  def iota(%T{shape: shape, type: type} = out, nil, _opts) do
    check_dtype!(type)
    w = worker()
    size = Nx.size(shape)

    r = Native.arange(w, 0.0, size * 1.0, 1.0, type)
    Native.reshape(w, r, shape_list(shape)) |> wrap(out, w)
  end

  def iota(%T{shape: shape, type: type} = out, axis, _opts) do
    check_dtype!(type)
    w = worker()
    dims = Tuple.to_list(shape)
    dim = Enum.at(dims, axis)

    line = Native.arange(w, 0.0, dim * 1.0, 1.0, type)

    thin_shape =
      dims
      |> Enum.with_index()
      |> Enum.map(fn {_, i} -> if i == axis, do: dim, else: 1 end)

    r = Native.reshape(w, line, thin_shape)
    Native.broadcast_to(w, r, dims) |> wrap(out, w)
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, _opts) do
    check_dtype!(type)
    w = worker()
    rank = tuple_size(shape)
    n = elem(shape, rank - 2)
    m = elem(shape, rank - 1)
    base = Native.eye(w, n, m, 0, type)

    if rank == 2 do
      wrap(base, out, w)
    else
      Native.broadcast_to(w, base, shape_list(shape)) |> wrap(out, w)
    end
  end

  # =================================================================
  # Cast
  # =================================================================

  @impl true
  def as_type(%T{type: type} = out, %T{} = t) do
    check_dtype!(type)
    w = worker()
    Native.astype(w, ref(t), type) |> wrap(out, w)
  end

  @impl true
  def bitcast(%T{type: type} = out, t) do
    w = worker()
    Native.bitcast(w, ref(t), type) |> wrap(out, w)
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
      w = worker()
      Native.unquote(native_name)(w, ref(t)) |> wrap(out, w)
    end
  end

  @direct_unary ~w(abs ceil floor sign sqrt rsqrt exp expm1 log log1p
                   sin cos tan sinh cosh tanh sigmoid erf conjugate
                   real imag)a

  for op <- @direct_unary do
    @impl true
    def unquote(op)(out, t) do
      w = worker()
      Native.unquote(op)(w, ref(t)) |> wrap(out, w)
    end
  end

  @impl true
  def round(out, t) do
    w = worker()
    Native.round(w, ref(t), 0) |> wrap(out, w)
  end

  @impl true
  def erfc(%T{type: type} = out, t) do
    w = worker()
    r = ref(t)
    one = scalar_ref(1, type)
    erf_r = Native.erf(w, r)
    Native.subtract(w, one, erf_r) |> wrap(out, w)
  end

  @impl true
  def cbrt(%T{type: type} = out, t) do
    w = worker()
    r = ref(t)
    signed = Native.sign(w, r)
    absolute = Native.abs(w, r)
    third = scalar_ref(1.0 / 3.0, type)
    pow = Native.power(w, absolute, third)
    Native.multiply(w, signed, pow) |> wrap(out, w)
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
  def logical_not(out, t) do
    w = worker()
    Native.logical_not(w, ref(t)) |> wrap(out, w)
  end

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
      w = worker()
      ra = Native.astype(w, ref(a), type)
      rb = Native.astype(w, ref(b), type)
      Native.unquote(native_name)(w, ra, rb) |> wrap(out, w)
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
      w = worker()
      target = Nx.Type.merge(a.type, b.type)
      ra = Native.astype(w, ref(a), target)
      rb = Native.astype(w, ref(b), target)
      Native.unquote(native_name)(w, ra, rb) |> wrap(out, w)
    end
  end

  @impl true
  def add(%T{type: type} = out, a, b) do
    w = worker()
    ra = Native.astype(w, ref(a), type)
    rb = Native.astype(w, ref(b), type)
    Native.add(w, ra, rb) |> wrap(out, w)
  end

  @impl true
  def quotient(%T{type: type} = out, a, b) do
    w = worker()
    ra = Native.astype(w, ref(a), type)
    rb = Native.astype(w, ref(b), type)
    Native.floor_divide(w, ra, rb) |> wrap(out, w)
  end

  @impl true
  def logical_xor(out, a, b) do
    w = worker()
    ra = ref(a)
    rb = ref(b)
    za = scalar_ref(0, a.type)
    zb = scalar_ref(0, b.type)
    ma = Native.not_equal(w, ra, za)
    mb = Native.not_equal(w, rb, zb)
    Native.not_equal(w, ma, mb) |> wrap(out, w)
  end

  # =================================================================
  # Shape
  # =================================================================

  @impl true
  def reshape(%T{shape: shape} = out, t) do
    w = worker()
    Native.reshape(w, ref(t), shape_list(shape)) |> wrap(out, w)
  end

  @impl true
  def squeeze(out, t, axes) do
    w = worker()
    Native.squeeze(w, ref(t), axes) |> wrap(out, w)
  end

  @impl true
  def transpose(out, t, axes) do
    w = worker()
    Native.transpose(w, ref(t), axes) |> wrap(out, w)
  end

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

    w = worker()

    r = Native.reshape(w, ref(t), intermediate)
    Native.broadcast_to(w, r, out_dims) |> wrap(out, w)
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

    w = worker()
    axes = Enum.to_list(0..(length(lows) - 1))
    Native.pad(w, ref(t), axes, lows, highs, ref(pad_value)) |> wrap(out, w)
  end

  @impl true
  def reverse(%T{} = out, t, axes) do
    w = worker()

    reversed =
      Enum.reduce(axes, ref(t), fn axis, acc ->
        Native.flip(w, acc, axis)
      end)

    wrap(reversed, out, w)
  end

  @impl true
  def concatenate(%T{} = out, tensors, axis) do
    w = worker()
    refs = Enum.map(tensors, &ref/1)
    Native.concatenate(w, refs, axis) |> wrap(out, w)
  end

  @impl true
  def stack(%T{} = out, tensors, axis) do
    w = worker()
    refs = Enum.map(tensors, &ref/1)
    Native.stack(w, refs, axis) |> wrap(out, w)
  end

  # =================================================================
  # Indexing
  # =================================================================

  @impl true
  def slice(%T{} = out, t, starts, lengths, strides) do
    # Nx passes starts as either integers or scalar tensors (dynamic
    # slicing). MLX's slice takes integer bounds; under the evaluator
    # we materialise scalar-tensor starts to their concrete value.
    w = worker()
    starts = Enum.map(starts, &slice_start/1)
    stops = Enum.zip_with(starts, lengths, fn st, l -> st + l end)
    Native.slice(w, ref(t), starts, stops, strides) |> wrap(out, w)
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
    w = worker()
    starts = Enum.map(start_indices, &slice_start/1)
    src_ref = Native.astype(w, ref(t), type)
    update_ref = Native.astype(w, ref(slice), type)
    Native.slice_update(w, src_ref, update_ref, starts) |> wrap(out, w)
  end

  @impl true
  # `{:pred, 1}` is Nx's 1-bit boolean dtype, mapped to `mx::bool_` in
  # `c_src/emily/dtype.hpp`. Not listed in `Nx.Type.t()` so
  # `Native.astype/3`'s spec is too narrow — suppress here rather than
  # widen the Native dtype type (that pollutes every wrapped tensor).
  @dialyzer {:nowarn_function, select: 4}
  def select(%T{} = out, pred, on_true, on_false) do
    w = worker()
    cond_ref = Native.astype(w, ref(pred), {:pred, 1})
    Native.where(w, cond_ref, ref(on_true), ref(on_false)) |> wrap(out, w)
  end

  @impl true
  def clip(%T{} = out, t, min_t, max_t) do
    w = worker()
    Native.clip(w, ref(t), ref(min_t), ref(max_t)) |> wrap(out, w)
  end

  # gather: Nx's gather takes a multi-dimensional index tensor whose
  # last dim selects across multiple axes. The single-axis case
  # (embedding lookups) uses Native.take; the multi-axis case uses
  # Native.gather (see `native_scatter_gather/0` docs).
  @impl true
  def gather(out, input, indices, opts) do
    axes = opts[:axes]
    indices_shape = Tuple.to_list(indices.shape)
    w = worker()

    cond do
      match?([_], axes) ->
        [axis] = axes
        idx_ref = Native.astype(w, ref(indices), {:s, 32})

        if @debug_bounds_check,
          do: DebugHelpers.check_bounds!(:gather, input.shape, [idx_ref], [axis], w)

        r = Native.take(w, ref(input), idx_ref, axis)
        Native.reshape(w, r, Tuple.to_list(out.shape)) |> wrap(out, w)

      scatter_gather_compatible?(indices_shape, axes) ->
        idx_refs = split_indices_per_axis(ref(indices), indices_shape, length(axes), w)

        if @debug_bounds_check,
          do: DebugHelpers.check_bounds!(:gather, input.shape, idx_refs, axes, w)

        slice_sizes = slice_sizes_for_gather(input.shape, axes)

        r = Native.gather(w, ref(input), idx_refs, axes, slice_sizes)
        Native.reshape(w, r, Tuple.to_list(out.shape)) |> wrap(out, w)

      true ->
        via_binary(:gather, out, [input, indices], &Nx.gather(&1, &2, opts))
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
      w = worker()
      axes = reduction_axes(opts, t)
      keep = opts[:keep_axes] || false
      Native.unquote(native_name)(w, ref(t), axes, keep) |> wrap(out, w)
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
    w = worker()
    axis = opts[:axis] || 0
    keep = tuple_size(out.shape) == tuple_size(t.shape)

    r = Native.argmax(w, ref(t), axis, keep)
    Native.astype(w, r, out.type) |> wrap(out, w)
  end

  @impl true
  def argmin(%T{} = out, t, opts) do
    w = worker()
    axis = opts[:axis] || 0
    keep = tuple_size(out.shape) == tuple_size(t.shape)

    r = Native.argmin(w, ref(t), axis, keep)
    Native.astype(w, r, out.type) |> wrap(out, w)
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
        w = worker()
        Native.unquote(native_name)(w, ref(t), axis, reverse, true) |> wrap(out, w)
      else
        nx_fun = unquote(nx_name)
        via_binary(nx_fun, out, [t], &apply(Nx, nx_fun, [&1, opts]))
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
    w = worker()
    Native.tensordot(w, ref(a), ref(b), contract_a, contract_b) |> wrap(out, w)
  end

  def dot(%T{type: type} = out, a, contract_a, batch_a, b, contract_b, batch_b) do
    # MLX matmul is float-only; ints/preds fall through to BinaryBackend.
    # In practice every transformer-attention call is float, so this is
    # the hot path.
    if float_like?(type) do
      batched_matmul(out, a, contract_a, batch_a, b, contract_b, batch_b)
    else
      via_binary(:dot, out, [a, b], &Nx.dot(&1, contract_a, batch_a, &2, contract_b, batch_b))
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

    w = worker()
    perm_a = batch_a ++ free_a ++ contract_a
    perm_b = batch_a ++ contract_b ++ free_b

    ra = Native.transpose(w, ref(a), perm_a)
    ra = Native.reshape(w, ra, [b_prod, m, k_prod])

    rb = Native.transpose(w, ref(b), perm_b)
    rb = Native.reshape(w, rb, [b_prod, k_prod, n])

    r = Native.matmul(w, ra, rb)
    if @debug_detect_nan_inf, do: DebugHelpers.check_nan_inf!(:matmul, r, w)
    Native.reshape(w, r, shape_list(out_shape)) |> wrap(out, w)
  end

  defp dim_product(axes, shape), do: Enum.reduce(axes, 1, &(elem(shape, &1) * &2))

  # =================================================================
  # Sort / argsort / top_k / all_close / take / take_along_axis
  # =================================================================

  @impl true
  def sort(%T{} = out, t, opts) do
    w = worker()
    axis = opts[:axis] || 0
    direction = opts[:direction] || :asc

    sorted = Native.sort(w, ref(t), axis)

    case direction do
      :asc -> wrap(sorted, out, w)
      :desc -> Native.flip(w, sorted, axis) |> wrap(out, w)
    end
  end

  @impl true
  def argsort(%T{} = out, t, opts) do
    w = worker()
    axis = opts[:axis] || 0
    direction = opts[:direction] || :asc

    idx = Native.argsort(w, ref(t), axis)

    idx =
      case direction do
        :asc -> idx
        :desc -> Native.flip(w, idx, axis)
      end

    Native.astype(w, idx, out.type) |> wrap(out, w)
  end

  # No `top_k/3` override. The real callback contract is
  # `top_k({out_values, out_indices}, tensor, opts) :: {values, indices}`,
  # but `mx::topk` only yields values. Without an override, Nx falls back
  # to `argsort(:desc) + take_along_axis + slice_along_axis`, all of
  # which route through MLX via this backend -- correct and no slower
  # than a handrolled implementation would be without a true dual-output
  # primitive.

  # all_close with absolute/relative tolerance:
  # all(abs(a - b) <= atol + rtol * abs(b)).
  @impl true
  def all_close(%T{} = out, a, b, opts) do
    w = worker()
    rtol = opts[:rtol] || 1.0e-5
    atol = opts[:atol] || 1.0e-8
    equal_nan = opts[:equal_nan] || false

    t = Nx.Type.merge(a.type, b.type) |> Nx.Type.to_floating()
    check_dtype!(t)

    ra = Native.astype(w, ref(a), t)
    rb = Native.astype(w, ref(b), t)

    diff = Native.abs(w, Native.subtract(w, ra, rb))

    tol =
      Native.add(
        w,
        scalar_ref(atol, t),
        Native.multiply(w, scalar_ref(rtol, t), Native.abs(w, rb))
      )

    close = Native.less_equal(w, diff, tol)

    close =
      if equal_nan do
        both_nan = Native.logical_and(w, Native.isnan(w, ra), Native.isnan(w, rb))
        Native.logical_or(w, close, both_nan)
      else
        close
      end

    axes = Enum.to_list(0..(tuple_size(a.shape) - 1))
    Native.all(w, close, axes, false) |> wrap(out, w)
  end

  @impl true
  def take(%T{} = out, input, indices, opts) do
    w = worker()
    axis = opts[:axis] || 0
    idx_ref = Native.astype(w, ref(indices), {:s, 32})

    if @debug_bounds_check,
      do: DebugHelpers.check_bounds!(:take, input.shape, [idx_ref], [axis], w)

    Native.take(w, ref(input), idx_ref, axis) |> wrap(out, w)
  end

  @impl true
  def take_along_axis(%T{} = out, input, indices, opts) do
    w = worker()
    axis = opts[:axis] || 0
    idx_ref = Native.astype(w, ref(indices), {:s, 32})

    if @debug_bounds_check,
      do: DebugHelpers.check_bounds!(:take_along_axis, input.shape, [idx_ref], [axis], w)

    Native.take_along_axis(w, ref(input), idx_ref, axis) |> wrap(out, w)
  end

  # =================================================================
  # FFT
  # =================================================================

  @impl true
  def fft(%T{} = out, t, opts) do
    w = worker()
    length = opts[:length]
    axis = tuple_size(t.shape) - 1
    Native.fftn(w, ref(t), [length], [axis]) |> wrap(out, w)
  end

  @impl true
  def ifft(%T{} = out, t, opts) do
    w = worker()
    length = opts[:length]
    axis = tuple_size(t.shape) - 1
    Native.ifftn(w, ref(t), [length], [axis]) |> wrap(out, w)
  end

  @impl true
  def fft2(%T{} = out, t, opts) do
    w = worker()
    lengths = opts[:lengths]
    axes = opts[:axes]
    Native.fftn(w, ref(t), lengths, axes) |> wrap(out, w)
  end

  @impl true
  def ifft2(%T{} = out, t, opts) do
    w = worker()
    lengths = opts[:lengths]
    axes = opts[:axes]
    Native.ifftn(w, ref(t), lengths, axes) |> wrap(out, w)
  end

  # =================================================================
  # Conv
  # =================================================================
  #
  # MLX `conv_general` expects NHWC input and OHWI weight; Nx's
  # canonical layout is NCHW / OIHW. Nx does not pre-transpose tensors
  # before dispatching — it delivers them in their original layout plus
  # `input_permutation`, `kernel_permutation`, `output_permutation`
  # such that `Nx.transpose(user_input, axes: input_permutation)` is
  # the canonical form. We transpose into NHWC/OHWI on the way in, call
  # the NIF, then reverse the layout on the way out.
  #
  # `batch_group_size > 1` and complex-typed conv have no MLX primitive
  # and fall back to `via_binary`; they are rare enough not to warrant
  # a handwritten MLX reshape trick.
  @impl true
  def conv(out, input, kernel, opts) do
    cond do
      opts[:batch_group_size] > 1 ->
        via_binary(:conv, out, [input, kernel], &Nx.conv(&1, &2, opts))

      match?({:c, _}, out.type) ->
        via_binary(:conv, out, [input, kernel], &Nx.conv(&1, &2, opts))

      true ->
        w = worker()
        ip = opts[:input_permutation]
        kp = opts[:kernel_permutation]
        op = opts[:output_permutation]
        {lows, highs} = opts[:padding] |> Enum.unzip()

        input_to_nhwc = [hd(ip)] ++ Enum.drop(ip, 2) ++ [Enum.at(ip, 1)]
        kernel_to_ohwi = [hd(kp)] ++ Enum.drop(kp, 2) ++ [Enum.at(kp, 1)]
        rank = tuple_size(out.shape)
        nhwc_to_nchw = [0, rank - 1] ++ Enum.to_list(1..(rank - 2)//1)
        inv_op = invert_permutation(op)

        ir = Native.transpose(w, Native.astype(w, ref(input), out.type), input_to_nhwc)
        kr = Native.transpose(w, Native.astype(w, ref(kernel), out.type), kernel_to_ohwi)

        conv_result =
          Native.conv_general(
            w,
            ir,
            kr,
            opts[:strides],
            {lows, highs},
            {opts[:kernel_dilation], opts[:input_dilation]},
            opts[:feature_group_size],
            false
          )

        conv_result
        |> then(&Native.transpose(w, &1, nhwc_to_nchw))
        |> then(&Native.transpose(w, &1, inv_op))
        |> wrap(out, w)
    end
  end

  # Invert a 0-based permutation: given `perm` where position i holds
  # j, produce `inv` where position j holds i. Used to reverse
  # `output_permutation` — Nx delivers it in "user → canonical" form
  # (see `deps/nx/lib/nx/shape.ex:729-735`), so we need the inverse to
  # go "canonical → user".
  defp invert_permutation(perm) do
    perm
    |> Enum.with_index()
    |> Enum.sort()
    |> Enum.map(&elem(&1, 1))
  end

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
  defp via_binary(op, %T{} = out, tensors, fun) when is_atom(op) and is_list(tensors) do
    metadata = fallback_metadata(op, tensors)
    Emily.Telemetry.maybe_warn_fallback(op, metadata.input_shapes)

    :telemetry.span([:emily, :fallback], metadata, fn ->
      result =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          tensors |> transfer_all() |> then(&apply(fun, &1))
        end)

      {from_binary(out, Nx.to_binary(result), []), metadata}
    end)
  end

  # Same pattern, but the op returns a tuple of tensors. `outs` is a
  # tuple of output templates matching arity; positions are zipped.
  defp via_binary_tuple(op, outs, tensors, fun)
       when is_atom(op) and is_tuple(outs) and is_list(tensors) do
    metadata = fallback_metadata(op, tensors)
    Emily.Telemetry.maybe_warn_fallback(op, metadata.input_shapes)

    :telemetry.span([:emily, :fallback], metadata, fn ->
      result_tuple =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          tensors |> transfer_all() |> then(&apply(fun, &1))
        end)

      result =
        outs
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(result_tuple))
        |> Enum.map(fn {out, r} -> from_binary(out, Nx.to_binary(r), []) end)
        |> List.to_tuple()

      {result, metadata}
    end)
  end

  defp fallback_metadata(op, tensors) do
    %{
      op: op,
      input_shapes: Enum.map(tensors, & &1.shape),
      input_dtypes: Enum.map(tensors, & &1.type)
    }
  end

  defp transfer_all(tensors),
    do: Enum.map(tensors, &Nx.backend_transfer(&1, Nx.BinaryBackend))

  @impl true
  def reduce(out, t, acc, opts, fun),
    do: via_binary(:reduce, out, [t, acc], &Nx.reduce(&1, &2, opts, fun))

  @impl true
  def window_reduce(out, t, acc, window_shape, opts, fun),
    do:
      via_binary(
        :window_reduce,
        out,
        [t, acc],
        &Nx.window_reduce(&1, &2, window_shape, opts, fun)
      )

  # M17: window reductions lifted off via_binary. MLX has no native
  # window_* primitive — each op is composed as pad → as_strided
  # (sliding-window view) → reduce in C++ (c_src/ops/pooling.cpp),
  # mirroring MLX's own nn/layers/pooling.py pattern.
  #
  # Nx passes padding already resolved to a list of `{lo, hi}` pairs
  # (Nx.Shape.pool runs upstream of the backend callback), plus per-axis
  # `:strides` and `:window_dilations`. We split the pairs and pass the
  # dtype-specific identity (0 for sum, 1 for product, ±∞ for max/min)
  # as the fill value — the reduce over the padded view then sees a
  # correct identity at the boundary.
  for {nx_name, native_name, identity} <- [
        {:window_sum, :window_sum, :zero},
        {:window_product, :window_product, :one},
        {:window_max, :window_max, :neg_inf},
        {:window_min, :window_min, :pos_inf}
      ] do
    @impl true
    def unquote(nx_name)(%T{} = out, t, window_shape, opts) do
      apply_window_reduce(
        out,
        t,
        window_shape,
        opts,
        unquote(native_name),
        unquote(identity)
      )
    end
  end

  # M17: window scatter variants lifted off via_binary — these are the
  # backward pass of window_max/window_min (Nx's grad rule rewrites
  # grad(window_max) into window_scatter_max), so lifting them is what
  # makes small-CNN training converge on MLX rather than spending every
  # backward in BinaryBackend.
  @impl true
  def window_scatter_max(%T{} = out, t, source, init, window_shape, opts) do
    apply_window_scatter(out, t, source, init, window_shape, opts, :window_scatter_max)
  end

  @impl true
  def window_scatter_min(%T{} = out, t, source, init, window_shape, opts) do
    apply_window_scatter(out, t, source, init, window_shape, opts, :window_scatter_min)
  end

  defp apply_window_reduce(%T{} = out, t, window_shape, opts, native_fun, identity) do
    w = worker()
    rank = tuple_size(t.shape)
    window = Tuple.to_list(window_shape)
    strides = normalize_per_axis(opts[:strides], rank, 1)
    dilations = normalize_per_axis(opts[:window_dilations], rank, 1)
    {pad_lo, pad_hi} = split_padding(opts[:padding], rank)
    init_ref = identity_ref(identity, t.type)

    apply(Native, native_fun, [w, ref(t), window, strides, pad_lo, pad_hi, dilations, init_ref])
    |> wrap(out, w)
  end

  defp apply_window_scatter(%T{} = out, t, source, init, window_shape, opts, native_fun) do
    w = worker()
    rank = tuple_size(t.shape)
    window = Tuple.to_list(window_shape)
    strides = normalize_per_axis(opts[:strides], rank, 1)
    {pad_lo, pad_hi} = split_padding(opts[:padding], rank)
    init_ref = coerce_scalar_ref(init, out.type, w)

    apply(Native, native_fun, [w, ref(t), ref(source), init_ref, window, strides, pad_lo, pad_hi])
    |> wrap(out, w)
  end

  # Resolve `:strides` / `:window_dilations` options to a length-rank
  # list. Nx normalises these upstream for the window ops (see
  # `Nx.aggregate_window_op/4`) so we rarely see bare integers, but we
  # stay defensive.
  defp normalize_per_axis(nil, rank, default), do: List.duplicate(default, rank)
  defp normalize_per_axis(n, rank, _default) when is_integer(n), do: List.duplicate(n, rank)
  defp normalize_per_axis(list, _rank, _default) when is_list(list), do: list

  # Split `[{lo, hi}, ...]` padding config into two lists. Nx always
  # resolves `:valid`/`:same` to per-axis pairs before the backend
  # callback, so we don't handle those atoms here.
  defp split_padding(pairs, _rank) when is_list(pairs) do
    pairs
    |> Enum.map(fn {lo, hi} -> {lo, hi} end)
    |> Enum.unzip()
  end

  defp split_padding(_, rank), do: {List.duplicate(0, rank), List.duplicate(0, rank)}

  defp identity_ref(:zero, type), do: scalar_ref(0, type)
  defp identity_ref(:one, type), do: scalar_ref(1, type)
  defp identity_ref(:neg_inf, {:f, _} = type), do: scalar_ref(:neg_infinity, type)
  defp identity_ref(:neg_inf, {:bf, _} = type), do: scalar_ref(:neg_infinity, type)

  defp identity_ref(:neg_inf, {kind, bits} = type) when kind in [:s, :u] do
    # Integer window_max: -inf doesn't exist, use the dtype's minimum.
    # u* minimum is 0; s* minimum is -(2^(bits-1)).
    value = if kind == :u, do: 0, else: -Bitwise.bsl(1, bits - 1)
    scalar_ref(value, type)
  end

  defp identity_ref(:pos_inf, {:f, _} = type), do: scalar_ref(:infinity, type)
  defp identity_ref(:pos_inf, {:bf, _} = type), do: scalar_ref(:infinity, type)

  defp identity_ref(:pos_inf, {kind, bits} = type) when kind in [:s, :u] do
    # Integer window_min: use the dtype's maximum.
    value =
      if kind == :u,
        do: Bitwise.bsl(1, bits) - 1,
        else: Bitwise.bsl(1, bits - 1) - 1

    scalar_ref(value, type)
  end

  # Coerce an already-provided scalar tensor (e.g. window_scatter's
  # `init_value`) to the target dtype and return its ref.
  defp coerce_scalar_ref(%T{} = init, type, w) do
    init_ref = ref(init)

    case Native.dtype(init_ref) do
      ^type -> init_ref
      _ -> Native.astype(w, init_ref, type)
    end
  end

  # indexed_add / indexed_put: Nx passes indices of shape {..., rank_of_axes}
  # and updates of shape {batch ++ non_indexed_dims}. MLX's scatter/scatter_add
  # take a list of per-axis index arrays and require updates to have rank
  # `indices[0].ndim() + target.ndim()`, with a length-1 dim inserted at
  # every indexed-axis position. We split the indices and rewrap updates
  # in Elixir (both operations are graph-construction — no data copied).
  #
  # Duplicate-index semantics differ: Nx.indexed_put is deterministic
  # last-write, MLX scatter is unordered. The grad test generators dedupe
  # indices; correctness on duplicates with indexed_put is best-effort.
  @impl true
  def indexed_add(out, t, indices, updates, opts) do
    apply_scatter(
      :indexed_add,
      out,
      t,
      indices,
      updates,
      opts,
      :scatter_add,
      &Nx.indexed_add(&1, &2, &3, opts)
    )
  end

  @impl true
  def indexed_put(out, t, indices, updates, opts) do
    apply_scatter(
      :indexed_put,
      out,
      t,
      indices,
      updates,
      opts,
      :scatter,
      &Nx.indexed_put(&1, &2, &3, opts)
    )
  end

  defp apply_scatter(op, out, t, indices, updates, opts, native_fun, fallback) do
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(t.shape) - 1))
    indices_shape = Tuple.to_list(indices.shape)

    if scatter_gather_compatible?(indices_shape, axes) do
      w = worker()
      idx_refs = split_indices_per_axis(ref(indices), indices_shape, length(axes), w)
      if @debug_bounds_check, do: DebugHelpers.check_bounds!(op, t.shape, idx_refs, axes, w)
      updates_shape = updates_shape_for_scatter(indices_shape, t.shape, axes)
      updates_ref = Native.reshape(w, ref(updates), updates_shape)

      apply(Native, native_fun, [w, ref(t), idx_refs, updates_ref, axes])
      |> wrap(out, w)
    else
      via_binary(op, out, [t, indices, updates], fallback)
    end
  end

  # Valid for native gather/scatter/scatter_add: indices tensor has a
  # trailing "rank of axes" dim, plus at least one batch dim, and axes
  # is a proper list.
  defp scatter_gather_compatible?(indices_shape, axes) do
    is_list(axes) and axes != [] and length(indices_shape) >= 2 and
      List.last(indices_shape) == length(axes)
  end

  # Split an {..., R} index tensor into R per-axis index tensors, each
  # of shape equal to the leading batch (last axis dropped).
  defp split_indices_per_axis(indices_ref, indices_shape, n_axes, w) do
    rank = length(indices_shape)
    last_axis = rank - 1
    batch_shape = Enum.take(indices_shape, last_axis)
    strides = List.duplicate(1, rank)
    batch_zeros = List.duplicate(0, last_axis)

    for i <- 0..(n_axes - 1) do
      sliced = Native.slice(w, indices_ref, batch_zeros ++ [i], batch_shape ++ [i + 1], strides)
      squeezed = Native.squeeze(w, sliced, [last_axis])
      Native.astype(w, squeezed, {:s, 32})
    end
  end

  # slice_sizes for MLX gather: length equal to rank(input); 1 on
  # indexed axes, full axis dim elsewhere.
  defp slice_sizes_for_gather(input_shape, axes) do
    axes_set = MapSet.new(axes)
    rank = tuple_size(input_shape)
    for i <- 0..(rank - 1), do: if(i in axes_set, do: 1, else: elem(input_shape, i))
  end

  # Rewrap Nx updates shape {batch ++ non_indexed_dims} to MLX's
  # required {batch ++ per_axis_slot}, where per_axis_slot has length
  # rank(target) with 1 on indexed axes and target_shape[i] on others.
  defp updates_shape_for_scatter(indices_shape, target_shape, axes) do
    batch = Enum.take(indices_shape, length(indices_shape) - 1)
    axes_set = MapSet.new(axes)
    rank = tuple_size(target_shape)
    trailing = for i <- 0..(rank - 1), do: if(i in axes_set, do: 1, else: elem(target_shape, i))
    batch ++ trailing
  end

  # =================================================================
  # Native linalg — decompositions & solvers via mx::linalg::*
  # =================================================================

  @impl true
  def lu({p_out, l_out, u_out}, t, _opts) do
    w = worker()
    {perm_ref, l_ref, u_ref} = Native.linalg_lu(w, ref(t))
    n = elem(t.shape, tuple_size(t.shape) - 1)
    eye_ref = Native.eye(w, n, n, 0, p_out.type)
    p_ref = Native.take(w, eye_ref, perm_ref, 0)
    {wrap(p_ref, p_out, w), wrap(l_ref, l_out, w), wrap(u_ref, u_out, w)}
  end

  @impl true
  def svd({u_out, s_out, v_out}, t, _opts) do
    w = worker()
    {u_ref, s_ref, v_ref} = Native.linalg_svd(w, ref(t))
    rank = tuple_size(t.shape)
    m = elem(t.shape, rank - 2)
    n = elem(t.shape, rank - 1)
    u_ref = maybe_slice_svd(u_ref, u_out.shape, {m, m}, w)
    v_ref = maybe_slice_svd(v_ref, v_out.shape, {n, n}, w)
    {wrap(u_ref, u_out, w), wrap(s_ref, s_out, w), wrap(v_ref, v_out, w)}
  end

  defp maybe_slice_svd(ref, out_shape, full_last2, w) do
    rank = tuple_size(out_shape)

    if {elem(out_shape, rank - 2), elem(out_shape, rank - 1)} == full_last2 do
      ref
    else
      starts = List.duplicate(0, rank)
      strides = List.duplicate(1, rank)
      Native.slice(w, ref, starts, Tuple.to_list(out_shape), strides)
    end
  end

  @impl true
  def triangular_solve(%T{} = out, a, b, opts) do
    w = worker()
    a_ref = ref(a)
    b_ref = ref(b)

    case {opts[:transform_a], opts[:left_side]} do
      {:none, true} ->
        Native.linalg_solve_triangular(w, a_ref, b_ref, not opts[:lower])
        |> wrap(out, w)

      {:transpose, true} ->
        at = Native.transpose(w, a_ref, mat_transpose_axes(a.shape))

        Native.linalg_solve_triangular(w, at, b_ref, opts[:lower])
        |> wrap(out, w)

      {:none, false} ->
        at = Native.transpose(w, a_ref, mat_transpose_axes(a.shape))
        bt = Native.transpose(w, b_ref, mat_transpose_axes(b.shape))
        xt = Native.linalg_solve_triangular(w, at, bt, opts[:lower])

        Native.transpose(w, xt, mat_transpose_axes(out.shape))
        |> wrap(out, w)

      {:transpose, false} ->
        bt = Native.transpose(w, b_ref, mat_transpose_axes(b.shape))
        xt = Native.linalg_solve_triangular(w, a_ref, bt, not opts[:lower])

        Native.transpose(w, xt, mat_transpose_axes(out.shape))
        |> wrap(out, w)
    end
  end

  defp mat_transpose_axes(shape) do
    rank = tuple_size(shape)
    Enum.to_list(0..(rank - 3)//1) ++ [rank - 1, rank - 2]
  end

  @impl true
  def qr({q_out, r_out}, t, opts) do
    case opts[:mode] do
      :reduced ->
        w = worker()
        {q_ref, r_ref} = Native.linalg_qr(w, ref(t))
        {wrap(q_ref, q_out, w), wrap(r_ref, r_out, w)}

      :complete ->
        via_binary_tuple(:qr, {q_out, r_out}, [t], &Nx.LinAlg.qr(&1, opts))
    end
  end

  @impl true
  def cholesky(%T{} = out, t) do
    w = worker()
    Native.linalg_cholesky(w, ref(t), false) |> wrap(out, w)
  end

  @impl true
  def eigh({vals_out, vecs_out}, t, _opts) do
    w = worker()
    {vals_ref, vecs_ref} = Native.linalg_eigh(w, ref(t), "L")
    {wrap(vals_ref, vals_out, w), wrap(vecs_ref, vecs_out, w)}
  end

  @impl true
  def solve(%T{} = out, a, b) do
    w = worker()
    Native.linalg_solve(w, ref(a), ref(b)) |> wrap(out, w)
  end

  # =================================================================
  # Custom fused-kernel callbacks for Emily.Fast
  # =================================================================
  #
  # These aren't part of the `Nx.Backend` behaviour — they're the
  # dispatch target for `Nx.Defn.Expr.optional/3` nodes emitted by
  # `Emily.Fast.*`. The Evaluator's `:optional` case looks up
  # `function_exported?(backend, op, arity)` and calls it with
  # `(out, args...)`. Non-Emily backends don't export these, so the
  # Evaluator runs the composed-defn fallback instead.
  #
  # Arities: one more than the `in_args` the `Emily.Fast.*` caller
  # passes (the leading `out` is the template tensor).

  @doc false
  def fast_rms_norm(%T{} = out, x, weight, opts) do
    w = worker()
    r = Native.fast_rms_norm(w, ref(x), ref(weight), opts[:eps] * 1.0)
    if @debug_detect_nan_inf, do: DebugHelpers.check_nan_inf!(:fast_rms_norm, r, w)
    wrap(r, out, w)
  end

  @doc false
  def fast_layer_norm(%T{} = out, x, weight, bias, opts) do
    w = worker()
    r = Native.fast_layer_norm(w, ref(x), ref(weight), ref(bias), opts[:eps] * 1.0)
    if @debug_detect_nan_inf, do: DebugHelpers.check_nan_inf!(:fast_layer_norm, r, w)
    wrap(r, out, w)
  end

  @doc false
  def fast_rope(%T{} = out, x, offset, opts) do
    w = worker()

    ref =
      Native.fast_rope(
        w,
        ref(x),
        opts[:dims],
        opts[:traditional],
        opts[:base] * 1.0,
        opts[:scale] * 1.0,
        ref(offset),
        nil
      )

    wrap(ref, out, w)
  end

  @doc false
  def fast_rope_with_freqs(%T{} = out, x, offset, freqs, opts) do
    w = worker()

    ref =
      Native.fast_rope(
        w,
        ref(x),
        opts[:dims],
        opts[:traditional],
        nil,
        opts[:scale] * 1.0,
        ref(offset),
        ref(freqs)
      )

    wrap(ref, out, w)
  end

  @doc false
  def fast_scaled_dot_product_attention(%T{} = out, q, k, v, opts) do
    w = worker()
    mask_mode = if opts[:causal], do: "causal", else: ""

    r =
      Native.fast_scaled_dot_product_attention(
        w,
        ref(q),
        ref(k),
        ref(v),
        opts[:scale] * 1.0,
        mask_mode,
        []
      )

    if @debug_detect_nan_inf,
      do: DebugHelpers.check_nan_inf!(:fast_scaled_dot_product_attention, r, w)

    wrap(r, out, w)
  end

  @doc false
  def fast_scaled_dot_product_attention_with_mask(%T{} = out, q, k, v, mask, opts) do
    w = worker()

    r =
      Native.fast_scaled_dot_product_attention(
        w,
        ref(q),
        ref(k),
        ref(v),
        opts[:scale] * 1.0,
        "array",
        [ref(mask)]
      )

    if @debug_detect_nan_inf,
      do: DebugHelpers.check_nan_inf!(:fast_scaled_dot_product_attention_with_mask, r, w)

    wrap(r, out, w)
  end
end
