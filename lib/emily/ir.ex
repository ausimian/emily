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
    any: 51
  }

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
    bin =
      number
      |> Nx.tensor(type: t.type, backend: Nx.BinaryBackend)
      |> Nx.broadcast(t.shape)
      |> Nx.to_binary()

    ref = Emily.Native.from_binary(bin, Tuple.to_list(t.shape), t.type)
    idx = state.n_consts
    {{:const, idx}, %{state | consts: [ref | state.consts], n_consts: idx + 1}}
  end

  defp lower_op(%T{data: %Nx.Defn.Expr{op: :tensor, args: [concrete]}}, state) do
    bin = Nx.to_binary(concrete)
    ref = Emily.Native.from_binary(bin, Tuple.to_list(concrete.shape), concrete.type)
    idx = state.n_captures
    {{:capture, idx}, %{state | captures: [ref | state.captures], n_captures: idx + 1}}
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

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op, args: [a]}}, state)
       when is_map_key(@unary_ops, op) do
    {ra, state} = lower_node(a, state)
    emit(state, Map.fetch!(@unary_ops, op), [ra])
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

  defp lower_op(%T{data: %Nx.Defn.Expr{op: op}}, _state) do
    raise ArgumentError,
          "Emily Expr compiler does not yet lower op #{inspect(op)} " <>
            "(no fallback). It will be added in a later milestone."
  end

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
end
