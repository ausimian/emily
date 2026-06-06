defmodule Emily.IR do
  @moduledoc false
  # Flat intermediate representation for the Expr->MLX single-NIF
  # compiler. A program is a topologically-ordered instruction list plus
  # binding tables for dynamic inputs, captured weights/constants, and
  # inline consts. `Emily.Program.compile/1` ships it to the
  # `compile_program` NIF, which builds a replayable program resource;
  # `Emily.Program.eval/4` then replays the whole graph in one round-trip.
  #
  # An operand reference is a tagged tuple `{kind, index}` where `kind`
  # selects the binding table:
  #
  #   * `{:input, i}`   — the i-th dynamic input (a `:parameter` node),
  #     supplied fresh per eval.
  #   * `{:capture, i}` — the i-th captured tensor (a baked weight /
  #     `:tensor` / `:constant`), held by the program resource and never
  #     re-shipped.
  #   * `{:const, i}`   — the i-th inline const tensor.
  #   * `{:instr, i}`   — the output of the i-th instruction. Must refer
  #     to a *prior* instruction (the program is a DAG in topo order).
  #
  # Refs are packed into an int64 (`pack_ref/1`) for the NIF boundary:
  # the high bits carry the kind, the low bits the index. Keep the
  # opcode table and ref encoding in lockstep with c_src/emily/opcodes.hpp
  # and c_src/emily/program.hpp.
  #
  # CM0 scope: the struct, ref packing, and the opcode table for `add`.
  # The `Nx.Defn.Expr` lowerer that *builds* an %Emily.IR{} from a traced
  # function lands in CM1 (this module grows a `lower/2`).

  import Bitwise

  @ref_tag_shift 48
  @index_mask (1 <<< @ref_tag_shift) - 1
  @max_index @index_mask

  # Opcode wire values — keep in sync with `enum class Opcode` in
  # c_src/emily/opcodes.hpp.
  @opcodes %{
    # binary arithmetic / bitwise
    add: 0,
    subtract: 1,
    multiply: 2,
    divide: 3,
    power: 4,
    maximum: 5,
    minimum: 6,
    remainder: 7,
    bitwise_and: 8,
    bitwise_or: 9,
    bitwise_xor: 10,
    left_shift: 11,
    right_shift: 12,
    # binary compare / logical
    equal: 13,
    not_equal: 14,
    less: 15,
    less_equal: 16,
    greater: 17,
    greater_equal: 18,
    logical_and: 19,
    logical_or: 20,
    # unary
    negative: 21,
    abs: 22,
    sign: 23,
    sqrt: 24,
    rsqrt: 25,
    square: 26,
    reciprocal: 27,
    exp: 28,
    log: 29,
    log1p: 30,
    sin: 31,
    cos: 32,
    tanh: 33,
    sigmoid: 34,
    floor: 35,
    ceil: 36,
    erf: 37,
    logical_not: 38,
    # cast / shape (carry iattrs)
    astype: 39,
    reshape: 40,
    transpose: 41,
    squeeze: 42,
    broadcast_to: 43,
    # linear algebra
    matmul: 44,
    tensordot: 45,
    # reductions (iattrs: [[axes...], [keepdims]])
    sum: 46,
    prod: 47,
    max: 48,
    min: 49,
    all: 50,
    any: 51,
    # indexing / selection
    where: 52,
    slice: 53,
    # fused transformer kernels (mx::fast::*); float attrs are int64 bit
    # patterns (float_bits/1).
    fast_rms_norm: 54,
    fast_layer_norm: 55,
    fast_rope: 56,
    fast_rope_freqs: 57,
    fast_sdpa: 58,
    fast_sdpa_mask: 59,
    quantized_matmul: 60,
    take: 61,
    concatenate: 62,
    dyn_slice_update: 63,
    conv_general: 64,
    # selection / sort (iattrs: argmax/argmin [[axis],[keepdims]]; sort/argsort [[axis]])
    argmax: 65,
    argmin: 66,
    clip: 67,
    sort: 68,
    argsort: 69,
    flip: 70,
    # control flow: operands = initial loop-carried state; iattrs [[arity]];
    # subprograms [condition, body]. Multi-output (produces `arity` values).
    while: 71,
    # RNG / dynamic indexing primitives
    bitcast: 72,
    erf_inv: 73,
    dyn_slice: 74,
    # inclusive cumulative reductions (iattrs [[axis],[reverse]])
    cumsum: 75,
    cumprod: 76,
    cummax: 77,
    cummin: 78,
    # multi-axis gather: operands [input, idx0, ...]; iattrs [[axes],[slice_sizes]]
    gather: 79,
    stack: 80,
    # take_along_axis: gather along one axis with a same-rank s32 index
    # tensor. operands [input, indices]; iattrs [[axis]].
    take_along_axis: 81,
    # window (pooling) reductions. operands [input, init_scalar]; iattrs
    # [[window],[strides],[pad_lo],[pad_hi],[dilations]].
    window_sum: 82,
    window_max: 83,
    window_min: 84,
    window_product: 85,
    # window select-and-scatter (pooling backward). operands
    # [input, source, init]; iattrs [[window],[strides],[pad_lo],[pad_hi]].
    window_scatter_max: 86,
    window_scatter_min: 87,
    # FFT family — n-D transforms. operands [input]; iattrs
    # [[sizes...],[axes...]]. `fft`/`ifft` (1-D, last axis) and the
    # `fft2`/`ifft2`/`rfft`/`irfft` blocks all route here. Unnormalized
    # (`FFTNorm::Backward`) is baked C++-side, matching Nx / the eager NIFs.
    fftn: 88,
    ifftn: 89,
    rfftn: 90,
    irfftn: 91,
    # Scatter (Nx.indexed_put / indexed_add). operands [target, updates,
    # idx0, ...] (one s32 index array per scattered axis); iattrs [[axes...]].
    # scatter overwrites (last-write on duplicates); scatter_add accumulates.
    scatter: 92,
    scatter_add: 93
  }

  # Quant mode string -> code; decoded by qmode_from_code in
  # c_src/emily/opcodes.hpp.
  @quant_modes %{"affine" => 0, "mxfp4" => 1, "mxfp8" => 2, "nvfp4" => 3}

  @ref_kinds %{input: 0, capture: 1, const: 2, instr: 3}
  @ref_kinds_inverse Map.new(@ref_kinds, fn {k, v} -> {v, k} end)

  # Nx dtype kind -> code; packed dtype code is `kind_code * 256 + bits`.
  # Keep in sync with `to_mlx_dtype_code` in c_src/emily/dtype.hpp.
  @dtype_kind_codes %{f: 0, bf: 1, s: 2, u: 3, c: 4, pred: 5}

  defstruct n_inputs: 0, captures: [], consts: [], instrs: [], outputs: []

  @type kind :: :input | :capture | :const | :instr
  @type ref :: {kind(), non_neg_integer()}
  @type instr :: %{
          required(:opcode) => atom(),
          required(:operands) => [ref()],
          optional(:iattrs) => [[integer()]]
        }
  @type t :: %__MODULE__{
          n_inputs: non_neg_integer(),
          captures: [Emily.Native.tensor()],
          consts: [Emily.Native.tensor()],
          instrs: [instr()],
          outputs: [ref()]
        }

  @doc """
  The opcode name -> wire-value map. Exposed for the opcode-parity test,
  which checks these stay in lockstep with the `Opcode` enum and
  `kOpcodeCount` in `c_src/emily/opcodes.hpp`.
  """
  @spec opcodes() :: %{atom() => non_neg_integer()}
  def opcodes, do: @opcodes

  @doc "Numeric wire value for an opcode name."
  @spec opcode(atom()) :: non_neg_integer()
  def opcode(name) when is_map_key(@opcodes, name), do: Map.fetch!(@opcodes, name)

  @doc "Whether `name` is a known opcode."
  @spec opcode?(atom()) :: boolean()
  def opcode?(name), do: is_map_key(@opcodes, name)

  @doc """
  Pack an Nx dtype `{kind, bits}` into the int code the `astype` opcode
  carries (`kind_code * 256 + bits`).
  """
  @spec dtype_code(Nx.Type.t()) :: non_neg_integer()
  def dtype_code({kind, bits}) when is_map_key(@dtype_kind_codes, kind),
    do: Map.fetch!(@dtype_kind_codes, kind) * 256 + bits

  @doc """
  Encode a float as the signed int64 bit pattern carried in `iattrs`
  (the IR's integer attribute channel). Decoded by `f64_from_bits` in
  c_src/emily/opcodes.hpp.
  """
  @spec float_bits(number()) :: integer()
  def float_bits(f) do
    <<bits::signed-64-native>> = <<f * 1.0::float-64-native>>
    bits
  end

  @doc "Pack a tagged operand ref into the int64 the NIF expects."
  @spec pack_ref(ref()) :: non_neg_integer()
  def pack_ref({kind, index})
      when is_map_key(@ref_kinds, kind) and is_integer(index) and index >= 0 and
             index <= @max_index do
    bor(bsl(Map.fetch!(@ref_kinds, kind), @ref_tag_shift), index)
  end

  @doc "Inverse of `pack_ref/1` — for tests and round-trip checks."
  @spec unpack_ref(non_neg_integer()) :: ref()
  def unpack_ref(packed) when is_integer(packed) and packed >= 0 do
    kind = Map.fetch!(@ref_kinds_inverse, bsr(packed, @ref_tag_shift) |> band(0x3))
    {kind, band(packed, @index_mask)}
  end

  # =================================================================
  # Lowering: Nx.Defn.Expr -> %Emily.IR{}
  # =================================================================
  #
  # Each Expr node's `op` is the `Nx.Backend` callback name and its
  # `args` mirror that callback's args (minus the output template), so
  # the per-op handling below ports `Emily.Backend`'s logic — including
  # the dtype-coercion casts it makes explicit — emitting IR
  # instructions instead of eager Native calls. Unsupported ops raise
  # (no silent fallback) per the compiler's no-fallback design.

  alias Emily.Fast.Block, as: FB
  alias Emily.Quantization.Block, as: QB
  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T

  # Nx Expr op -> IR opcode. Arithmetic/bitwise cast both operands to the
  # node's out.type before the op (see backend.ex @renamed_arith_binary).
  @arith_binary %{
    add: :add,
    subtract: :subtract,
    multiply: :multiply,
    divide: :divide,
    pow: :power,
    remainder: :remainder,
    min: :minimum,
    max: :maximum,
    bitwise_and: :bitwise_and,
    bitwise_or: :bitwise_or,
    bitwise_xor: :bitwise_xor,
    left_shift: :left_shift,
    right_shift: :right_shift
  }

  # Compare/logical cast both operands to Nx.Type.merge(a, b); the op
  # yields MLX bool, coerced to the node's out.type ({:u, 8}).
  @compare_binary %{
    equal: :equal,
    not_equal: :not_equal,
    less: :less,
    less_equal: :less_equal,
    greater: :greater,
    greater_equal: :greater_equal,
    logical_and: :logical_and,
    logical_or: :logical_or
  }

  # Unary elementwise: no coercion (MLX preserves the dtype Nx expects).
  @unary_ops %{
    negate: :negative,
    abs: :abs,
    sign: :sign,
    sqrt: :sqrt,
    rsqrt: :rsqrt,
    exp: :exp,
    log: :log,
    log1p: :log1p,
    sin: :sin,
    cos: :cos,
    tanh: :tanh,
    sigmoid: :sigmoid,
    floor: :floor,
    ceil: :ceil,
    erf: :erf,
    erf_inv: :erf_inv
  }

  @doc """
  Lower a list of output-leaf `Nx.Defn.Expr` tensors into an `%Emily.IR{}`.

  The returned IR's `outputs` are in the same order as `output_leaves`,
  so the caller can reassemble its output container by zipping the
  program's results back in order. Captured weights / materialized
  constants are held as native tensor refs in `captures` / `consts`.
  """
  @spec lower([T.t()]) :: t()
  def lower(output_leaves) when is_list(output_leaves) do
    state = %{
      cache: %{},
      instrs: [],
      n_instrs: 0,
      captures: [],
      n_captures: 0,
      consts: [],
      n_consts: 0,
      n_inputs: 0
    }

    {output_refs, state} = Enum.map_reduce(output_leaves, state, &lower_node/2)

    %__MODULE__{
      n_inputs: state.n_inputs,
      captures: Enum.reverse(state.captures),
      consts: Enum.reverse(state.consts),
      instrs: Enum.reverse(state.instrs),
      outputs: output_refs
    }
  end

  defp lower_node(%T{data: %Nx.Defn.Expr{id: id}} = t, state) do
    case state.cache do
      %{^id => ref} ->
        {ref, state}

      _ ->
        {ref, state} = lower_op(t, state)
        {ref, %{state | cache: Map.put(state.cache, id, ref)}}
    end
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}}, state) do
    {{:input, pos}, %{state | n_inputs: max(state.n_inputs, pos + 1)}}
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :constant, args: [number]}} = t, state) do
    number
    |> Nx.tensor(type: t.type, backend: Nx.BinaryBackend)
    |> Nx.broadcast(t.shape)
    |> materialize_const(t.shape, t.type, state)
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :tensor, args: [concrete]}}, state) do
    materialize_capture(concrete, concrete.shape, concrete.type, state)
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :metadata, args: [expr, _meta]}}, state) do
    lower_node(expr, state)
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, b]}} = t, state)
       when is_map_key(@arith_binary, op) do
    {ra, state} = lower_node(a, state)
    {rb, state} = lower_node(b, state)
    code = dtype_code(t.type)
    {ca, state} = emit(state, :astype, [ra], [[code]])
    {cb, state} = emit(state, :astype, [rb], [[code]])
    emit(state, Map.fetch!(@arith_binary, op), [ca, cb])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, b]}} = t, state)
       when is_map_key(@compare_binary, op) do
    {ra, state} = lower_node(a, state)
    {rb, state} = lower_node(b, state)
    merged = dtype_code(Nx.Type.merge(a.type, b.type))
    {ca, state} = emit(state, :astype, [ra], [[merged]])
    {cb, state} = emit(state, :astype, [rb], [[merged]])
    {r, state} = emit(state, Map.fetch!(@compare_binary, op), [ca, cb])
    emit(state, :astype, [r], [[dtype_code(t.type)]])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a]}} = t, state)
       when is_map_key(@unary_ops, op) do
    {ra, state} = lower_node(a, state)
    {r, state} = emit(state, Map.fetch!(@unary_ops, op), [ra])
    # Coerce to out.type to match Emily.Backend.wrap/3 (every eager op is
    # wrapped through coerce). MLX unary dtype usually equals Nx's
    # out.type, so this is a no-op astype, but it keeps the node's dtype
    # exact regardless of MLX promotion.
    coerce(r, t.type, state)
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :as_type, args: [a]}} = t, state) do
    {ra, state} = lower_node(a, state)
    emit(state, :astype, [ra], [[dtype_code(t.type)]])
  end

  # bitcast: reinterpret the bytes as out.type (mirrors Emily.Backend.bitcast/2,
  # which calls mx::view). Used by the RNG path to turn random bits into floats.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :bitcast, args: [a]}} = t, state) do
    {ra, state} = lower_node(a, state)
    emit(state, :bitcast, [ra], [[dtype_code(t.type)]])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :reshape, args: [a]}} = t, state) do
    {ra, state} = lower_node(a, state)
    emit(state, :reshape, [ra], [Tuple.to_list(t.shape)])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :squeeze, args: [a, axes]}}, state) do
    {ra, state} = lower_node(a, state)
    emit(state, :squeeze, [ra], [axes])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :transpose, args: [a, axes]}}, state) do
    {ra, state} = lower_node(a, state)
    emit(state, :transpose, [ra], [axes])
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :broadcast, args: [a, _shape, axes]}} = t, state) do
    {ra, state} = lower_node(a, state)
    in_dims = Tuple.to_list(a.shape)
    out_dims = Tuple.to_list(t.shape)
    placed = Enum.zip(axes, in_dims) |> Map.new()

    intermediate =
      out_dims |> Enum.with_index() |> Enum.map(fn {_, i} -> Map.get(placed, i, 1) end)

    {rr, state} = emit(state, :reshape, [ra], [intermediate])
    emit(state, :broadcast_to, [rr], [out_dims])
  end

  # Reductions (sum/product/all/any/reduce_max/reduce_min): args
  # [tensor, opts]. opts carries :axes (nil = all) and :keep_axes.
  @reductions %{
    sum: :sum,
    product: :prod,
    all: :all,
    any: :any,
    reduce_max: :max,
    reduce_min: :min
  }

  # Cumulative reductions arrive as `Nx.block/4` nodes. Block struct -> opcode.
  @cumulative_blocks %{
    Nx.Block.CumulativeSum => :cumsum,
    Nx.Block.CumulativeProduct => :cumprod,
    Nx.Block.CumulativeMax => :cummax,
    Nx.Block.CumulativeMin => :cummin
  }

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, opts]}} = t, state)
       when is_map_key(@reductions, op) do
    {ra, state} = lower_node(a, state)
    axes = opts[:axes] || Nx.axes(a)
    keep = if(opts[:keep_axes], do: 1, else: 0)
    {r, state} = emit(state, Map.fetch!(@reductions, op), [ra], [axes, [keep]])
    coerce(r, t.type, state)
  end

  # argmax / argmin: args [tensor, opts]. axis defaults to 0; keepdims is
  # derived from whether the output kept the reduced axis (mirrors
  # Emily.Backend.argmax/3, then a coerce to the integer out.type). MLX
  # argmax/argmin is first-occurrence, matching Nx's default `tie_break:
  # :low`; like the backend we don't special-case `:tie_break`.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, opts]}} = t, state)
       when op in [:argmax, :argmin] do
    {ra, state} = lower_node(a, state)
    axis = opts[:axis] || 0
    keep = if(tuple_size(t.shape) == tuple_size(a.shape), do: 1, else: 0)
    {r, state} = emit(state, op, [ra], [[axis], [keep]])
    coerce(r, t.type, state)
  end

  # clip(t, min, max): element-wise clamp. Mirrors Emily.Backend.clip/4 —
  # the three operands pass straight to mx::clip, result coerced to out.type.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :clip, args: [a, min, max]}} = t, state) do
    {ra, state} = lower_node(a, state)
    {rmin, state} = lower_node(min, state)
    {rmax, state} = lower_node(max, state)
    emit_coerced(state, :clip, [ra, rmin, rmax], [], t.type)
  end

  # sort / argsort: args [tensor, opts] with :axis and :direction. MLX sorts
  # ascending; `:desc` reverses along the axis with a negative-stride slice —
  # exactly Emily.Backend.{sort,argsort}/3's `Native.flip`. The trailing
  # coerce is a no-op astype for sort (dtype-preserving) and the index cast
  # for argsort.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, opts]}} = t, state)
       when op in [:sort, :argsort] do
    {ra, state} = lower_node(a, state)
    axis = opts[:axis] || 0
    {r, state} = emit(state, op, [ra], [[axis]])

    {r, state} =
      case opts[:direction] || :asc do
        :asc -> {r, state}
        :desc -> emit(state, :flip, [r], [[axis]])
      end

    coerce(r, t.type, state)
  end

  # 1-D FFT / inverse FFT (Nx.fft / Nx.ifft). Mirrors Emily.Backend.{fft,
  # ifft}/3: route through the n-D MLX kernel restricted to one axis. The
  # eager path uses the trailing axis and ignores `opts[:axis]`, so we do
  # too — keeping native bit-identical to the evaluator. Output is complex
  # (`Nx.Type.to_complex/1`); the trailing coerce matches the backend `wrap`.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, opts]}} = t, state)
       when op in [:fft, :ifft] do
    {ra, state} = lower_node(a, state)
    axis = tuple_size(a.shape) - 1
    opcode = if op == :fft, do: :fftn, else: :ifftn
    emit_coerced(state, opcode, [ra], [[opts[:length]], [axis]], t.type)
  end

  # Non-batched dot -> tensordot over the contraction axes.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :dot, args: [a, ca, [], b, cb, []]}} = t, state) do
    {ra, state} = lower_node(a, state)
    {rb, state} = lower_node(b, state)
    {r, state} = emit(state, :tensordot, [ra, rb], [ca, cb])
    coerce(r, t.type, state)
  end

  # Batched dot -> permute to [batch, free, contract] / [batch, contract,
  # free], flatten to 3-D, MLX matmul (leading dims = batch), reshape to
  # Nx's `batch ++ free_a ++ free_b`. Mirrors Emily.Backend.batched_matmul/7.
  # MLX matmul is float-only; non-float batched dot raises (no fallback).
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: :dot, args: [a, ca, batch_a, b, cb, _batch_b]}} = t,
         state
       ) do
    unless float_like?(t.type) do
      raise ArgumentError,
            "Emily Expr compiler: batched dot on #{inspect(t.type)} is not supported " <>
              "(MLX matmul is float-only, and the compiler does not fall back)."
    end

    as = a.shape
    bs = b.shape
    a_rank = tuple_size(as)
    b_rank = tuple_size(bs)
    k = length(batch_a)

    contract_set_a = MapSet.new(ca)
    contract_set_b = MapSet.new(cb)
    free_a = for i <- k..(a_rank - 1)//1, not MapSet.member?(contract_set_a, i), do: i
    free_b = for i <- k..(b_rank - 1)//1, not MapSet.member?(contract_set_b, i), do: i

    b_prod = dim_product(batch_a, as)
    m = dim_product(free_a, as)
    n = dim_product(free_b, bs)
    k_prod = dim_product(ca, as)

    {ra, state} = lower_node(a, state)
    {rb, state} = lower_node(b, state)

    {ra, state} = emit(state, :transpose, [ra], [batch_a ++ free_a ++ ca])
    {ra, state} = emit(state, :reshape, [ra], [[b_prod, m, k_prod]])
    {rb, state} = emit(state, :transpose, [rb], [batch_a ++ cb ++ free_b])
    {rb, state} = emit(state, :reshape, [rb], [[b_prod, k_prod, n]])
    {r, state} = emit(state, :matmul, [ra, rb])
    {r, state} = emit(state, :reshape, [r], [Tuple.to_list(t.shape)])
    coerce(r, t.type, state)
  end

  # select(pred, on_true, on_false): cast pred to {:pred, 1}, then where.
  # Mirrors Emily.Backend.select/4.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :select, args: [pred, on_true, on_false]}} = t, state) do
    {rp, state} = lower_node(pred, state)
    {rt, state} = lower_node(on_true, state)
    {rf, state} = lower_node(on_false, state)
    {rp, state} = emit(state, :astype, [rp], [[dtype_code({:pred, 1})]])
    {r, state} = emit(state, :where, [rp, rt, rf])
    coerce(r, t.type, state)
  end

  # slice(t, starts, lengths, strides). Nx passes starts as integers (static
  # slice) or scalar tensors (dynamic slice — e.g. threefry indexing a
  # rotation table by the loop counter). Static starts -> mx::slice with
  # integer bounds. Dynamic starts -> mx::slice's dynamic-start overload via
  # the `dyn_slice` opcode (stride 1 only; the eager backend materialises the
  # start to a host int, but the compiled replay can't, so it threads the
  # start as a runtime s32 array).
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: :slice, args: [a, starts, lengths, strides]}} = t,
         state
       ) do
    {ra, state} = lower_node(a, state)

    if Enum.all?(starts, &is_integer/1) do
      stops = Enum.zip_with(starts, lengths, fn st, l -> st + l end)
      {r, state} = emit(state, :slice, [ra], [starts, stops, strides])
      coerce(r, t.type, state)
    else
      unless Enum.all?(strides, &(&1 == 1)) do
        raise ArgumentError,
              "Emily Expr compiler: dynamic (tensor) slice start indices are only " <>
                "supported with unit strides. Got strides: #{inspect(strides)}"
      end

      # Build the [ndim] s32 start array from the mixed int / scalar-tensor
      # starts (same machinery as the dynamic put_slice write). Each runtime
      # start is clamped to `[0, dim - length]` — MLX's dynamic slice reads
      # out of bounds, whereas Nx (XLA semantics) clamps the start so the
      # window stays in range; clamp is a no-op for the in-bounds starts the
      # threefry/RNG path produces.
      dims = Tuple.to_list(a.shape)

      {start_refs, state} =
        [starts, lengths, dims]
        |> Enum.zip()
        |> Enum.map_reduce(state, fn {start, length, dim}, state ->
          {r, state} = lower_start_index(start, state)
          clamp_start(r, dim - length, state)
        end)

      {start_arr, state} = emit(state, :concatenate, start_refs, [[0]])
      axes = Enum.to_list(0..(length(starts) - 1)//1)
      emit_coerced(state, :dyn_slice, [ra, start_arr], [axes, lengths], t.type)
    end
  end

  # gather(input, indices, opts). Mirrors Emily.Backend.gather/4: single-axis
  # gathers `take` along the one axis (indices cast to s32); multi-axis
  # gathers split the `{..., R}` index tensor into R per-axis index arrays
  # and use MLX's multi-index gather. Both reshape to the output shape (token
  # selection in sampling is this shape). A layout MLX gather can't express
  # raises, so the graceful fallback handles it.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :gather, args: [input, indices, opts]}} = t, state) do
    axes = opts[:axes]
    indices_shape = Tuple.to_list(indices.shape)

    {ri, state} = lower_node(input, state)
    {rx, state} = lower_node(indices, state)

    {r, state} =
      case axes do
        [axis] ->
          {ix, state} = emit(state, :astype, [rx], [[dtype_code({:s, 32})]])
          emit(state, :take, [ri, ix], [[axis]])

        _ when is_list(axes) ->
          unless scatter_gather_compatible?(indices_shape, axes) do
            raise ArgumentError,
                  "Emily Expr compiler: gather index layout #{inspect(indices_shape)} " <>
                    "for axes #{inspect(axes)} is not MLX-gather-compatible."
          end

          {idx_refs, state} = split_indices_for_gather(rx, indices_shape, length(axes), state)
          slice_sizes = slice_sizes_for_gather(input.shape, axes)
          emit(state, :gather, [ri | idx_refs], [axes, slice_sizes])
      end

    {r, state} = emit(state, :reshape, [r], [Tuple.to_list(t.shape)])
    coerce(r, t.type, state)
  end

  # indexed_put / indexed_add (Nx scatter). Mirrors Emily.Backend's
  # apply_scatter: split the {..., R} index tensor into R per-axis s32 index
  # arrays, reshape `updates` into MLX's scatter layout, then mx::scatter
  # (overwrite) / mx::scatter_add (accumulate). Only MLX-scatter-compatible
  # index layouts lower; others raise (no fallback) — the Evaluator handles
  # them via_binary under `native_fallback: :eval`, matching native gather.
  # Operands [target, updates, idx0, ...]; iattrs [[axes...]].
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: op, args: [target, indices, updates, opts]}} = t,
         state
       )
       when op in [:indexed_put, :indexed_add] do
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(target.shape) - 1)//1)
    indices_shape = Tuple.to_list(indices.shape)

    unless scatter_gather_compatible?(indices_shape, axes) do
      raise ArgumentError,
            "Emily Expr compiler: #{op} index layout #{inspect(indices_shape)} for axes " <>
              "#{inspect(axes)} is not MLX-scatter-compatible (no fallback)."
    end

    {rt, state} = lower_node(target, state)
    {rx, state} = lower_node(indices, state)
    {idx_refs, state} = split_indices_for_gather(rx, indices_shape, length(axes), state)

    {ru, state} = lower_node(updates, state)
    updates_shape = updates_shape_for_scatter(indices_shape, target.shape, axes)
    {ru, state} = emit(state, :reshape, [ru], [updates_shape])

    opcode = if op == :indexed_put, do: :scatter, else: :scatter_add
    emit_coerced(state, opcode, [rt, ru | idx_refs], [axes], t.type)
  end

  # put_slice(src, start_indices, slice): write `slice` into `src` at
  # `start_indices`. Mirrors Emily.Backend.put_slice/4 (cast src + update
  # to out.type), but supports RUNTIME (tensor) start indices — the decode
  # KV write at a dynamic offset. Builds an s32 start array from the mixed
  # int / scalar-tensor entries and uses MLX's dynamic slice_update.
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: :put_slice, args: [src, start_indices, slice]}} = t,
         state
       ) do
    type = t.type
    {rsrc, state} = lower_node(src, state)
    {rsrc, state} = emit(state, :astype, [rsrc], [[dtype_code(type)]])
    {rupd, state} = lower_node(slice, state)
    {rupd, state} = emit(state, :astype, [rupd], [[dtype_code(type)]])

    # One s32 [1] scalar per axis; concatenate into the [ndim] start array.
    {start_refs, state} = Enum.map_reduce(start_indices, state, &lower_start_index/2)
    {start_arr, state} = emit(state, :concatenate, start_refs, [[0]])

    # All put_slice routes through the dynamic slice_update kernel (start
    # indices may be runtime inputs). For all-static starts the Evaluator
    # uses the static kernel instead, but the two are bit-identical for the
    # in-bounds writes Nx.put_slice produces.
    axes = Enum.to_list(0..(tuple_size(t.shape) - 1)//1)
    emit_coerced(state, :dyn_slice_update, [rsrc, rupd, start_arr], [axes], type)
  end

  # iota: a pure creation op (shape/axis/type all static), so materialize
  # it as a captured constant instead of adding a runtime opcode.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :iota, args: [axis]}} = t, state) do
    t.shape
    |> Nx.iota(axis: axis, type: t.type, backend: Nx.BinaryBackend)
    |> materialize_const(t.shape, t.type, state)
  end

  # Nx.block node: args [struct, in_args, expr, callback]. Known fused
  # structs lower to their fused opcode (matching Emily.Backend.block/4,
  # which the Evaluator dispatches through); unknown structs lower by
  # recursing into `expr` — the pre-composed default expansion into
  # primitives — so there is still no runtime fallback.
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: :block, args: [struct, in_args, expr, _cb]}} = t,
         state
       ) do
    lower_block(struct, in_args, expr, t, state)
  end

  # concatenate(tensors, axis): join a list of tensors along `axis`.
  # Mirrors Emily.Backend.concatenate/3 (no input cast; the result is
  # coerced to out.type).
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :concatenate, args: [tensors, axis]}} = t, state) do
    {refs, state} = Enum.map_reduce(tensors, state, &lower_node/2)
    emit_coerced(state, :concatenate, refs, [[axis]], t.type)
  end

  # stack(tensors, axis): join along a NEW axis. Mirrors Emily.Backend.stack/3.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :stack, args: [tensors, axis]}} = t, state) do
    {refs, state} = Enum.map_reduce(tensors, state, &lower_node/2)
    emit_coerced(state, :stack, refs, [[axis]], t.type)
  end

  # conv: ports Emily.Backend.conv/4 — permute input -> NHWC and kernel ->
  # OHWI (casting both to out.type), mx::conv_general, then permute the
  # result NHWC -> NCHW -> the user's output layout. batch_group_size > 1
  # and complex types are unsupported (the backend falls back; we raise —
  # no fallback).
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :conv, args: [input, kernel, opts]}} = t, state) do
    if opts[:batch_group_size] > 1 or match?({:c, _}, t.type) do
      raise ArgumentError,
            "Emily Expr compiler: conv with batch_group_size > 1 or a complex " <>
              "output type is not supported (no fallback)."
    end

    type = t.type
    ip = opts[:input_permutation]
    kp = opts[:kernel_permutation]
    {lows, highs} = opts[:padding] |> Enum.unzip()

    input_to_nhwc = [hd(ip)] ++ Enum.drop(ip, 2) ++ [Enum.at(ip, 1)]
    kernel_to_ohwi = [hd(kp)] ++ Enum.drop(kp, 2) ++ [Enum.at(kp, 1)]
    rank = tuple_size(t.shape)
    nhwc_to_nchw = [0, rank - 1] ++ Enum.to_list(1..(rank - 2)//1)
    inv_op = invert_permutation(opts[:output_permutation])

    {ir, state} = lower_node(input, state)
    {ir, state} = emit(state, :astype, [ir], [[dtype_code(type)]])
    {ir, state} = emit(state, :transpose, [ir], [input_to_nhwc])

    {kr, state} = lower_node(kernel, state)
    {kr, state} = emit(state, :astype, [kr], [[dtype_code(type)]])
    {kr, state} = emit(state, :transpose, [kr], [kernel_to_ohwi])

    conv_attrs = [
      opts[:strides],
      lows,
      highs,
      opts[:kernel_dilation],
      opts[:input_dilation],
      [opts[:feature_group_size]],
      [0]
    ]

    {r, state} = emit(state, :conv_general, [ir, kr], conv_attrs)
    {r, state} = emit(state, :transpose, [r], [nhwc_to_nchw])
    {r, state} = emit(state, :transpose, [r], [inv_op])
    coerce(r, type, state)
  end

  # Window (pooling) reductions (window_sum/max/min/product). Mirrors
  # Emily.Backend.apply_window_reduce/6: pad with the dtype identity, build
  # the sliding-window view, reduce over the kernel axes. The init scalar
  # is baked as a const operand so float ±inf and integer min/max are all
  # handled exactly as the eager `identity_ref`. Operands [input, init];
  # iattrs [[window],[strides],[pad_lo],[pad_hi],[dilations]].
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, window_dimensions, opts]}} = t, state)
       when op in [:window_sum, :window_max, :window_min, :window_product] do
    {ra, state} = lower_node(a, state)
    {init_ref, state} = window_identity(op, a.type, state)

    rank = tuple_size(a.shape)
    window = Tuple.to_list(window_dimensions)
    strides = window_per_axis(opts[:strides], rank, 1)
    dilations = window_per_axis(opts[:window_dilations], rank, 1)
    {pad_lo, pad_hi} = window_padding(opts[:padding], rank)

    emit_coerced(state, op, [ra, init_ref], [window, strides, pad_lo, pad_hi, dilations], t.type)
  end

  # Window select-and-scatter (window_scatter_max/min) — the MaxPool/MinPool
  # backward. Mirrors Emily.Backend.apply_window_scatter/7: operands
  # [input, source, init]; iattrs [[window],[strides],[pad_lo],[pad_hi]]
  # (no dilations). The init scalar comes from the Expr (Nx's grad rule),
  # coerced to the output dtype.
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: op, args: [tin, source, init, window_dimensions, opts]}} = t,
         state
       )
       when op in [:window_scatter_max, :window_scatter_min] do
    {rt, state} = lower_node(tin, state)
    {rs, state} = lower_node(source, state)
    {ri, state} = lower_node(init, state)
    {ri, state} = emit(state, :astype, [ri], [[dtype_code(t.type)]])

    rank = tuple_size(tin.shape)
    window = Tuple.to_list(window_dimensions)
    strides = window_per_axis(opts[:strides], rank, 1)
    {pad_lo, pad_hi} = window_padding(opts[:padding], rank)

    emit_coerced(state, op, [rt, rs, ri], [window, strides, pad_lo, pad_hi], t.type)
  end

  # Nx.reverse along one or more axes (the conv backward flips the kernel).
  # Reversing is order-independent across axes, so chain a single-axis
  # `flip` (mx negative-stride slice) per axis. Empty axes => identity.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :reverse, args: [a, axes]}}, state) do
    {ra, state} = lower_node(a, state)
    Enum.reduce(axes, {ra, state}, fn axis, {r, st} -> emit(st, :flip, [r], [[axis]]) end)
  end

  # cond: raw args [clauses, last], clauses = [{pred, body}, ...]. Lower to
  # a select chain `where(p1, b1, where(p2, b2, ... last))`. ALL branches
  # are evaluated (Nx branches are side-effect-free and shape-compatible);
  # the result value matches the Evaluator's chosen branch exactly — only
  # the cost differs (not-taken branches are computed and discarded by the
  # elementwise select). The predicate is a whole-tensor scalar bool, so
  # `where` selects a branch wholesale.
  #
  # Caveat: a not-taken branch is still computed. On MLX an out-of-bounds
  # gather/index there clamps rather than faults, so the discarded value
  # never changes the result; a hard-faulting op on a not-taken path would
  # diverge from the Evaluator's lazy single-branch eval.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :cond, args: [clauses, %T{} = last]}} = t, state) do
    {last_ref, state} = lower_node(last, state)

    {result, state} =
      Enum.reduce(Enum.reverse(clauses), {last_ref, state}, fn {pred, body}, {else_ref, st} ->
        {pred_ref, st} = lower_node(pred, st)
        {body_ref, st} = lower_node(body, st)
        {pred_ref, st} = emit(st, :astype, [pred_ref], [[dtype_code({:pred, 1})]])
        emit(st, :where, [pred_ref, body_ref, else_ref])
      end)

    coerce(result, t.type, state)
  end

  # Multi-output cond: each branch returns a tuple of tensors. Lower to one
  # `where`-chain per leaf position — identical wholesale-select semantics to
  # the single-output case above (the predicate is a whole-tensor scalar bool;
  # every branch is still computed). Returns a `{:multi_refs, [...]}` handle
  # that `:elem` projects; sibling `:elem`s share it via lower_node's memo.
  # A nested / non-tensor container raises (no fallback path for it yet).
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :cond, args: [clauses, last]}}, state)
       when is_tuple(last) do
    last_leaves = Tuple.to_list(last)

    unless tensors?(last_leaves) and
             Enum.all?(clauses, fn {_p, b} -> is_tuple(b) and tensors?(Tuple.to_list(b)) end) do
      raise ArgumentError,
            "Emily Expr compiler: cond over a nested / non-tensor container is not " <>
              "lowered yet (only a flat tuple of tensors)."
    end

    # Lower each predicate (cast to pred) + its branch's leaf refs once, so the
    # per-leaf where-chains share them.
    {clauses, state} =
      Enum.map_reduce(clauses, state, fn {pred, body}, st ->
        {pred_ref, st} = lower_node(pred, st)
        {pred_ref, st} = emit(st, :astype, [pred_ref], [[dtype_code({:pred, 1})]])
        {body_refs, st} = Enum.map_reduce(Tuple.to_list(body), st, &lower_node/2)
        {{pred_ref, body_refs}, st}
      end)

    rev = Enum.reverse(clauses)

    {refs, state} =
      last_leaves
      |> Enum.with_index()
      |> Enum.map_reduce(state, fn {leaf, j}, st ->
        {last_ref, st} = lower_node(leaf, st)

        {result, st} =
          Enum.reduce(rev, {last_ref, st}, fn {pred_ref, body_refs}, {else_ref, st2} ->
            emit(st2, :where, [pred_ref, Enum.at(body_refs, j), else_ref])
          end)

        coerce(result, leaf.type, st)
      end)

    {{:multi_refs, refs}, state}
  end

  # attach_token: sequences a token (hooks) before `expr`. With no active
  # hook the token is a no-op, so pass through to the inner expr. Hooks
  # would need a callback into Elixir mid-graph (program-split) — deferred.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :attach_token, args: [token, expr]}}, state) do
    if Tree.has_hooks?(token, %{}) do
      raise ArgumentError,
            "Emily Expr compiler does not support hooks under native compilation " <>
              "(they require a mid-graph callback into Elixir)."
    end

    lower_node(expr, state)
  end

  # reduce / window_reduce with a user-supplied BEAM reducer cannot be
  # compiled — the reducer would have to run on the host mid-graph. The
  # fixed-identity aggregates (sum/product/max/min) are separate ops and
  # already lower natively; only an arbitrary reducer reaches here.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op}}, _state) when op in [:reduce, :window_reduce] do
    raise ArgumentError,
          "Emily Expr compiler cannot lower #{inspect(op)} with an arbitrary " <>
            "reducer function (it would require a host callback mid-graph; no " <>
            "fallback). Use the native aggregates (sum/product/reduce_max/" <>
            "reduce_min) where possible."
  end

  # while: args [flatten_initial, flatten_arg, condition, flatten_body] (see
  # Nx.Defn.Expr.while/5). Each `flatten_*` is a single tensor or a tuple of
  # tensors (the flattened loop-carried state). The condition and body are
  # expressions over the `flatten_arg` parameter nodes.
  #
  # We lower the initial state in the *parent* program, then lower the
  # condition and body into their own sub-programs whose inputs are the
  # loop-carried state: each `flatten_arg` leaf is pre-bound to `{:input, i}`,
  # so at replay the worker binds the current state vector as the
  # subprogram's inputs. The `while` instruction reserves `arity` value slots
  # (its outputs are the final state); `:elem` projects them.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :while, args: [initial, arg, cond, body]}} = t, state) do
    init_leaves = flatten_leaves(initial)
    arg_leaves = flatten_leaves(arg)
    body_leaves = flatten_leaves(body)
    arity = length(init_leaves)

    {init_refs, state} = Enum.map_reduce(init_leaves, state, &lower_node/2)

    param_seed =
      arg_leaves
      |> Enum.with_index()
      |> Map.new(fn {%T{data: %Nx.Defn.Expr{id: id}}, i} -> {id, {:input, i}} end)

    cond_ir = lower_subgraph([cond], param_seed, arity)
    body_ir = lower_subgraph(body_leaves, param_seed, arity)

    {base, state} = emit_multi(state, :while, init_refs, [[arity]], [cond_ir, body_ir], arity)

    # A tuple-typed `while` is consumed only through `:elem`; return the
    # multi-output handle for those to project. A single-tensor `while`
    # (arity 1, not tuple-typed) is used directly — return its one ref.
    case t.type do
      {:tuple, _} -> {{:multi, base, arity}, state}
      _ -> {{:instr, base}, state}
    end
  end

  # :elem projects the i-th output of a multi-output instruction (today,
  # `while`). lower_node memoizes the producer, so sibling `:elem`s share one
  # `while` instruction.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :elem, args: [tuple_expr, i]}}, state) do
    case lower_node(tuple_expr, state) do
      {{:multi, base, _arity}, state} ->
        {{:instr, base + i}, state}

      {{:multi_refs, refs}, state} ->
        {Enum.at(refs, i), state}

      {_handle, _state} ->
        raise ArgumentError,
              "Emily Expr compiler: :elem projects a tuple-producing op it can't " <>
                "lower yet (only `while` and tuple `cond` produce projectable " <>
                "tuples today; other multi-output ops are unsupported)."
    end
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError,
          "Emily Expr compiler does not yet lower op #{inspect(op)} " <>
            "(no fallback). It will be added in a later milestone."
  end

  defp lower_block(%FB.RMSNorm{eps: eps}, [x, weight], _expr, t, state) do
    {rx, state} = lower_node(x, state)
    {rw, state} = lower_node(weight, state)
    emit_coerced(state, :fast_rms_norm, [rx, rw], [[float_bits(eps)]], t.type)
  end

  defp lower_block(%FB.LayerNorm{eps: eps}, [x, weight, bias], _expr, t, state) do
    {rx, state} = lower_node(x, state)
    {rw, state} = lower_node(weight, state)
    {rb, state} = lower_node(bias, state)
    emit_coerced(state, :fast_layer_norm, [rx, rw, rb], [[float_bits(eps)]], t.type)
  end

  defp lower_block(%FB.RoPE{} = b, [x, offset], _expr, t, state) do
    {rx, state} = lower_node(x, state)
    {ro, state} = lower_node(offset, state)

    attrs = [[b.dims], [bool_int(b.traditional)], [float_bits(b.base)], [float_bits(b.scale)]]
    emit_coerced(state, :fast_rope, [rx, ro], attrs, t.type)
  end

  defp lower_block(%FB.RoPEWithFreqs{} = b, [x, offset, freqs], _expr, t, state) do
    {rx, state} = lower_node(x, state)
    {ro, state} = lower_node(offset, state)
    {rf, state} = lower_node(freqs, state)

    attrs = [[b.dims], [bool_int(b.traditional)], [float_bits(b.scale)]]
    emit_coerced(state, :fast_rope_freqs, [rx, ro, rf], attrs, t.type)
  end

  defp lower_block(%FB.SDPA{scale: scale, causal: causal}, [q, k, v], _expr, t, state) do
    {rq, state} = lower_node(q, state)
    {rk, state} = lower_node(k, state)
    {rv, state} = lower_node(v, state)

    emit_coerced(
      state,
      :fast_sdpa,
      [rq, rk, rv],
      [[float_bits(scale)], [bool_int(causal)]],
      t.type
    )
  end

  defp lower_block(%FB.SDPAWithMask{scale: scale}, [q, k, v, mask], _expr, t, state) do
    {rq, state} = lower_node(q, state)
    {rk, state} = lower_node(k, state)
    {rv, state} = lower_node(v, state)
    {rm, state} = lower_node(mask, state)
    emit_coerced(state, :fast_sdpa_mask, [rq, rk, rv, rm], [[float_bits(scale)]], t.type)
  end

  defp lower_block(%QB.QuantizedMatmul{} = qb, [x, q, s, b], _expr, t, state) do
    {rx, state} = lower_node(x, state)
    {rq, state} = lower_node(q, state)
    {rs, state} = lower_node(s, state)
    {rb, state} = lower_node(b, state)

    attrs = [
      [bool_int(qb.transpose)],
      [qb.group_size],
      [qb.bits],
      [Map.fetch!(@quant_modes, qb.mode)]
    ]

    emit_coerced(state, :quantized_matmul, [rx, rq, rs, rb], attrs, t.type)
  end

  # Nx.take / embedding lookup (Nx.Block.Take). Mirrors
  # Emily.Backend.native_take/4: cast indices to s32, then mx::take.
  defp lower_block(%Nx.Block.Take{axis: axis}, [input, indices], _expr, t, state) do
    {ri, state} = lower_node(input, state)
    {rx, state} = lower_node(indices, state)
    {rx, state} = emit(state, :astype, [rx], [[dtype_code({:s, 32})]])
    emit_coerced(state, :take, [ri, rx], [[axis]], t.type)
  end

  # Nx.take_along_axis (Nx.Block.TakeAlongAxis). Mirrors
  # Emily.Backend.native_take_along_axis/4: cast indices to s32, then
  # mx::take_along_axis along `axis`.
  defp lower_block(%Nx.Block.TakeAlongAxis{axis: axis}, [input, indices], _expr, t, state) do
    {ri, state} = lower_node(input, state)
    {rx, state} = lower_node(indices, state)
    {rx, state} = emit(state, :astype, [rx], [[dtype_code({:s, 32})]])
    emit_coerced(state, :take_along_axis, [ri, rx], [[axis]], t.type)
  end

  # Cumulative families. Like Emily.Backend.block/4, the last-axis case uses
  # the native MLX `cumsum`/`cumprod`/`cummax`/`cummin` kernel; interior axes
  # (which MLX can't always factor) fall back to the block's composed
  # expansion. Nx cumulation is always inclusive.
  defp lower_block(%mod{axis: axis, reverse: reverse}, [t], expr, out, state)
       when is_map_key(@cumulative_blocks, mod) do
    if axis == tuple_size(out.shape) - 1 do
      {rt, state} = lower_node(t, state)
      op = Map.fetch!(@cumulative_blocks, mod)
      emit_coerced(state, op, [rt], [[axis], [bool_int(reverse)]], out.type)
    else
      lower_node(expr, state)
    end
  end

  # FFT family blocks (Nx.fft2 / ifft2 / rfft / irfft). Each mirrors the
  # matching Emily.Backend.native_* wrapper: route through the n-D MLX
  # fft/ifft/rfft/irfft kernel with the block's sizes + axes, then coerce to
  # out.type (complex for the forward transforms, real for irfft). The
  # block's `eps` is unused (MLX needs none), as in the eager path.
  defp lower_block(%Nx.Block.FFT2{lengths: lengths, axes: axes}, [t], _expr, out, state) do
    {rt, state} = lower_node(t, state)
    emit_coerced(state, :fftn, [rt], [lengths, axes], out.type)
  end

  defp lower_block(%Nx.Block.IFFT2{lengths: lengths, axes: axes}, [t], _expr, out, state) do
    {rt, state} = lower_node(t, state)
    emit_coerced(state, :ifftn, [rt], [lengths, axes], out.type)
  end

  defp lower_block(%Nx.Block.RFFT{length: length, axis: axis}, [t], _expr, out, state) do
    {rt, state} = lower_node(t, state)
    emit_coerced(state, :rfftn, [rt], [[length], [axis]], out.type)
  end

  defp lower_block(%Nx.Block.IRFFT{length: length, axis: axis}, [t], _expr, out, state) do
    {rt, state} = lower_node(t, state)
    emit_coerced(state, :irfftn, [rt], [[length], [axis]], out.type)
  end

  # Any other block struct raises. Lowering the block's composed
  # expansion would silently diverge from the Evaluator whenever
  # Emily.Backend.block/4 dispatches that struct through a fused / native
  # kernel (e.g. SDPAWithSinks, the Nx.Block.LinAlg.* families) — a
  # worse failure than a clear "unsupported".
  # Additional fused blocks are added alongside their opcode.
  defp lower_block(struct, _in_args, _expr, _t, _state) do
    raise ArgumentError,
          "Emily Expr compiler does not yet lower the block " <>
            "#{inspect(struct.__struct__)} (no fallback). Supported: RMSNorm, " <>
            "LayerNorm, RoPE, RoPEWithFreqs, SDPA, SDPAWithMask, QuantizedMatmul."
  end

  # The flattened loop-carried state of a `while` arg: Nx delivers it as a
  # single tensor (single-tensor state) or a tuple of tensors.
  defp flatten_leaves(%T{} = t), do: [t]
  defp flatten_leaves(tuple) when is_tuple(tuple), do: Tuple.to_list(tuple)

  # Lower `output_leaves` into a self-contained sub-IR whose inputs are the
  # loop-carried state. `param_seed` maps each loop parameter's Expr id to
  # `{:input, i}`, pre-seeded into the cache so those nodes resolve to the
  # bound state rather than being treated as captures. The sub-IR carries its
  # own captures/consts (e.g. weights the body closes over) and `n_inputs`
  # = arity. Replayed by the C++ `while` arm with the current state as inputs.
  defp lower_subgraph(output_leaves, param_seed, arity) do
    state = %{
      cache: param_seed,
      instrs: [],
      n_instrs: 0,
      captures: [],
      n_captures: 0,
      consts: [],
      n_consts: 0,
      n_inputs: arity
    }

    {output_refs, state} = Enum.map_reduce(output_leaves, state, &lower_node/2)

    %__MODULE__{
      n_inputs: arity,
      captures: Enum.reverse(state.captures),
      consts: Enum.reverse(state.consts),
      instrs: Enum.reverse(state.instrs),
      outputs: output_refs
    }
  end

  # Append a multi-output instruction (carrying sub-programs) producing
  # `arity` values. Only one entry joins `instrs`, but `n_instrs` advances by
  # `arity` to reserve the output slots — the i-th is `{:instr, base + i}` —
  # so subsequent refs stay aligned with the `values` vector the C++ replay
  # builds (its `while` arm pushes `arity` results for this one instruction).
  defp emit_multi(state, opcode, operands, iattrs, subprograms, arity) do
    base = state.n_instrs
    instr = %{opcode: opcode, operands: operands, iattrs: iattrs, subprograms: subprograms}
    {base, %{state | instrs: [instr | state.instrs], n_instrs: state.n_instrs + arity}}
  end

  defp bool_int(true), do: 1
  defp bool_int(false), do: 0

  defp tensors?(list), do: Enum.all?(list, &match?(%T{}, &1))

  defp float_like?({kind, _}) when kind in [:f, :bf, :c], do: true
  defp float_like?(_), do: false

  # Invert a 0-based permutation (mirrors Emily.Backend.invert_permutation/1)
  # — reverse Nx's "user -> canonical" output_permutation to "canonical ->
  # user" for the final conv transpose.
  defp invert_permutation(perm) do
    perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  defp dim_product(axes, shape), do: Enum.reduce(axes, 1, &(elem(shape, &1) * &2))

  # Coerce a ref to `type` (emit an astype). MLX astype to the same dtype
  # is a no-op, so this is safe to apply unconditionally — it mirrors
  # Emily.Backend.wrap/3's coerce and keeps the node's dtype exact.
  defp coerce(ref, type, state), do: emit(state, :astype, [ref], [[dtype_code(type)]])

  # Append an instruction, returning its {:instr, i} ref.
  defp emit(state, opcode, operands, iattrs \\ []) do
    ref = {:instr, state.n_instrs}
    instr = %{opcode: opcode, operands: operands, iattrs: iattrs}
    {ref, %{state | instrs: [instr | state.instrs], n_instrs: state.n_instrs + 1}}
  end

  # Emit an instruction then coerce its output to `type` (mirrors
  # Emily.Backend.wrap/3). The trailing coerce is mandatory on every
  # value-producing op, so the helper keeps the per-clause tail honest.
  defp emit_coerced(state, opcode, operands, iattrs, type) do
    {r, state} = emit(state, opcode, operands, iattrs)
    coerce(r, type, state)
  end

  # Materialize an Nx tensor (already on a host backend) as a captured
  # const / weight ref, held by the program for its lifetime. `:const`
  # holds materialized literal constants (and iota); `:capture` holds
  # embedded `:tensor` weights. Both go through from_binary once at
  # lower time and are never re-shipped.
  # Bake the dtype identity for a window reduce as a scalar const operand
  # (mirrors Emily.Backend.identity_ref/2): 0 for sum, 1 for product,
  # ±inf for float max/min, dtype min/max for integer max/min.
  defp window_identity(op, type, state) do
    materialize_const(window_identity_scalar(op, type), {}, type, state)
  end

  defp window_identity_scalar(:window_sum, type),
    do: Nx.tensor(0, type: type, backend: Nx.BinaryBackend)

  defp window_identity_scalar(:window_product, type),
    do: Nx.tensor(1, type: type, backend: Nx.BinaryBackend)

  defp window_identity_scalar(:window_max, {kind, _} = type) when kind in [:f, :bf],
    do: Nx.tensor(:neg_infinity, type: type, backend: Nx.BinaryBackend)

  defp window_identity_scalar(:window_min, {kind, _} = type) when kind in [:f, :bf],
    do: Nx.tensor(:infinity, type: type, backend: Nx.BinaryBackend)

  defp window_identity_scalar(:window_max, {kind, bits} = type) when kind in [:s, :u] do
    value = if kind == :u, do: 0, else: -Bitwise.bsl(1, bits - 1)
    Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
  end

  defp window_identity_scalar(:window_min, {kind, bits} = type) when kind in [:s, :u] do
    value = if kind == :u, do: Bitwise.bsl(1, bits) - 1, else: Bitwise.bsl(1, bits - 1) - 1
    Nx.tensor(value, type: type, backend: Nx.BinaryBackend)
  end

  # Per-axis strides / dilations (mirror Emily.Backend.normalize_per_axis/3).
  defp window_per_axis(nil, rank, default), do: List.duplicate(default, rank)
  defp window_per_axis(n, rank, _default) when is_integer(n), do: List.duplicate(n, rank)
  defp window_per_axis(list, _rank, _default) when is_list(list), do: list

  # Split resolved `[{lo, hi}, ...]` padding into two lists (mirror
  # Emily.Backend.split_padding/2).
  defp window_padding(pairs, _rank) when is_list(pairs) do
    pairs |> Enum.map(fn {lo, hi} -> {lo, hi} end) |> Enum.unzip()
  end

  defp window_padding(_other, rank),
    do: {List.duplicate(0, rank), List.duplicate(0, rank)}

  defp materialize_const(tensor, shape, type, state) do
    ref = Emily.Native.from_binary(Nx.to_binary(tensor), Tuple.to_list(shape), type)
    idx = state.n_consts
    {{:const, idx}, %{state | consts: [ref | state.consts], n_consts: idx + 1}}
  end

  defp materialize_capture(tensor, shape, type, state) do
    ref = Emily.Native.from_binary(Nx.to_binary(tensor), Tuple.to_list(shape), type)
    idx = state.n_captures
    {{:capture, idx}, %{state | captures: [ref | state.captures], n_captures: idx + 1}}
  end

  # One s32 `[1]` start index per put_slice axis. Integer starts become a
  # const `[1]` scalar (the Expr path delivers static starts as :constant
  # `%T{}` nodes, so this clause is mainly for hand-built IR); scalar-tensor
  # (dynamic) starts are cast to s32 and reshaped to `[1]`. The caller
  # concatenates them into the `[ndim]` start array MLX's dynamic
  # slice_update consumes.
  defp lower_start_index(i, state) when is_integer(i) do
    materialize_const(Nx.tensor([i], type: :s32, backend: Nx.BinaryBackend), {1}, {:s, 32}, state)
  end

  defp lower_start_index(%T{} = expr, state) do
    {r, state} = lower_node(expr, state)
    {r, state} = emit(state, :astype, [r], [[dtype_code({:s, 32})]])
    emit(state, :reshape, [r], [[1]])
  end

  # Clamp a runtime s32 `[1]` dynamic-slice start to `[0, hi]`
  # (hi = dim - length), matching Nx/XLA dynamic-slice semantics — MLX's
  # dynamic slice would otherwise read out of bounds. Both bounds are static.
  defp clamp_start(ref, hi, state) do
    {lo_c, state} =
      materialize_const(
        Nx.tensor([0], type: :s32, backend: Nx.BinaryBackend),
        {1},
        {:s, 32},
        state
      )

    {hi_c, state} =
      materialize_const(
        Nx.tensor([hi], type: :s32, backend: Nx.BinaryBackend),
        {1},
        {:s, 32},
        state
      )

    {r, state} = emit(state, :maximum, [ref, lo_c])
    emit(state, :minimum, [r, hi_c])
  end

  # MLX's multi-index gather needs the index tensor's leading dims to be the
  # batch and the last axis to select across `axes` (mirrors
  # Emily.Backend.scatter_gather_compatible?/2).
  defp scatter_gather_compatible?(indices_shape, axes) do
    is_list(axes) and axes != [] and length(indices_shape) >= 2 and
      List.last(indices_shape) == length(axes)
  end

  # Split an `{..., R}` index tensor into R per-axis s32 index arrays (each
  # the leading batch with the last axis dropped) — ports
  # Emily.Backend.split_indices_per_axis/4 with static slices.
  defp split_indices_for_gather(indices_ref, indices_shape, n_axes, state) do
    rank = length(indices_shape)
    last_axis = rank - 1
    batch_shape = Enum.take(indices_shape, last_axis)
    strides = List.duplicate(1, rank)
    batch_zeros = List.duplicate(0, last_axis)

    Enum.map_reduce(0..(n_axes - 1)//1, state, fn i, state ->
      {r, state} =
        emit(state, :slice, [indices_ref], [batch_zeros ++ [i], batch_shape ++ [i + 1], strides])

      {r, state} = emit(state, :squeeze, [r], [[last_axis]])
      emit(state, :astype, [r], [[dtype_code({:s, 32})]])
    end)
  end

  # Per-axis slice size for gather: 1 on a gathered axis, the full extent
  # otherwise (mirrors Emily.Backend.slice_sizes_for_gather/2).
  defp slice_sizes_for_gather(input_shape, axes) do
    axes_set = MapSet.new(axes)
    rank = tuple_size(input_shape)
    for i <- 0..(rank - 1)//1, do: if(i in axes_set, do: 1, else: elem(input_shape, i))
  end

  # Rewrap Nx's updates shape {batch ++ non_indexed_dims} into MLX's scatter
  # layout {batch ++ per_axis_slot}, where per_axis_slot has length
  # rank(target) with 1 on indexed axes and target_shape[i] elsewhere. Mirrors
  # Emily.Backend.updates_shape_for_scatter/3.
  defp updates_shape_for_scatter(indices_shape, target_shape, axes) do
    batch = Enum.take(indices_shape, length(indices_shape) - 1)
    axes_set = MapSet.new(axes)
    rank = tuple_size(target_shape)

    trailing =
      for i <- 0..(rank - 1)//1, do: if(i in axes_set, do: 1, else: elem(target_shape, i))

    batch ++ trailing
  end
end
