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
    dyn_slice_update: 63
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
    erf: :erf
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

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a, opts]}} = t, state)
       when is_map_key(@reductions, op) do
    {ra, state} = lower_node(a, state)
    axes = opts[:axes] || Nx.axes(a)
    keep = if(opts[:keep_axes], do: 1, else: 0)
    {r, state} = emit(state, Map.fetch!(@reductions, op), [ra], [axes, [keep]])
    coerce(r, t.type, state)
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

  # slice(t, starts, lengths, strides): static integer starts only. Nx
  # passes scalar-tensor starts for dynamic slicing; those depend on
  # runtime values and are deferred (the decode offset becomes a runtime
  # input in CM3). Stops = starts + lengths (see Emily.Backend.slice/5).
  defp lower_op(
         %T{data: %Nx.Defn.Expr{op: :slice, args: [a, starts, lengths, strides]}} = t,
         state
       ) do
    unless Enum.all?(starts, &is_integer/1) do
      raise ArgumentError,
            "Emily Expr compiler: dynamic (tensor) slice start indices are not yet " <>
              "supported (they require a runtime input). Got: #{inspect(starts)}"
    end

    {ra, state} = lower_node(a, state)
    stops = Enum.zip_with(starts, lengths, fn st, l -> st + l end)
    {r, state} = emit(state, :slice, [ra], [starts, stops, strides])
    coerce(r, t.type, state)
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

  # cond: raw args [clauses, last], clauses = [{pred, body}, ...]. Lower to
  # a select chain `where(p1, b1, where(p2, b2, ... last))`. ALL branches
  # are evaluated (Nx branches are side-effect-free and shape-compatible);
  # the result value matches the Evaluator's chosen branch exactly — only
  # the cost differs (not-taken branches are computed and discarded). The
  # predicate is a whole-tensor scalar bool, so `where` selects a branch
  # wholesale.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: :cond, args: [clauses, last]}} = t, state) do
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

  # while / its tuple projection are deferred to a follow-up: the
  # single-NIF replay has no loop construct, so a data-dependent while
  # needs either static-trip unrolling or a worker-side synced loop.
  defp lower_op(%T{data: %Nx.Defn.Expr{op: op}}, _state) when op in [:while, :elem] do
    raise ArgumentError,
          "Emily Expr compiler does not yet lower #{inspect(op)} (defn `while` " <>
            "loops; deferred — the decode/generation loops run in Elixir today)."
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

  # Any other block struct raises. Lowering the block's composed
  # expansion would silently diverge from the Evaluator whenever
  # Emily.Backend.block/4 dispatches that struct through a fused / native
  # kernel (e.g. SDPAWithSinks, the Nx.Block.LinAlg.* / Take / FFT /
  # cumulative families) — a worse failure than a clear "unsupported".
  # Additional fused blocks are added alongside their opcode.
  defp lower_block(struct, _in_args, _expr, _t, _state) do
    raise ArgumentError,
          "Emily Expr compiler does not yet lower the block " <>
            "#{inspect(struct.__struct__)} (no fallback). Supported: RMSNorm, " <>
            "LayerNorm, RoPE, RoPEWithFreqs, SDPA, SDPAWithMask, QuantizedMatmul."
  end

  defp bool_int(true), do: 1
  defp bool_int(false), do: 0

  defp float_like?({kind, _}) when kind in [:f, :bf, :c], do: true
  defp float_like?(_), do: false

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
end
