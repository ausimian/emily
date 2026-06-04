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
    broadcast_to: 43
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
end
