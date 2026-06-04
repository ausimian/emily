defmodule Emily.IRTest do
  @moduledoc """
  CM0 unit tests for the Expr-compiler flat IR: operand-ref
  packing/unpacking and the compile -> describe round-trip (the C++ side
  stores the IR faithfully). The Native oracle here is "what we shipped
  comes back unchanged"; no GPU work beyond building tiny capture tensors.
  """
  use ExUnit.Case, async: true

  import Emily.TensorHelpers

  alias Emily.{IR, Program}

  describe "operand-ref packing" do
    test "pack_ref/unpack_ref round-trips every kind" do
      for ref <- [{:input, 0}, {:capture, 3}, {:const, 7}, {:instr, 12}] do
        assert IR.unpack_ref(IR.pack_ref(ref)) == ref
      end
    end

    test "distinct kinds at the same index pack to distinct ints" do
      packed = Enum.map([:input, :capture, :const, :instr], &IR.pack_ref({&1, 5}))
      assert length(Enum.uniq(packed)) == 4
    end

    test "large indices survive the round-trip" do
      ref = {:instr, 1_000_000}
      assert IR.unpack_ref(IR.pack_ref(ref)) == ref
    end

    test "pack_ref rejects an unknown kind or negative index" do
      assert_raise FunctionClauseError, fn -> IR.pack_ref({:bogus, 0}) end
      assert_raise FunctionClauseError, fn -> IR.pack_ref({:input, -1}) end
    end
  end

  describe "compile -> describe round-trip" do
    test "single add instruction" do
      ir = %IR{
        n_inputs: 2,
        instrs: [%{opcode: :add, operands: [{:input, 0}, {:input, 1}]}],
        outputs: [{:instr, 0}]
      }

      prog = Program.compile(ir)
      {n_inputs, n_captures, n_consts, opcodes, operands, outputs} = Program.describe(prog)

      assert n_inputs == 2
      assert n_captures == 0
      assert n_consts == 0
      assert opcodes == [IR.opcode(:add)]
      assert operands == [Enum.map([{:input, 0}, {:input, 1}], &IR.pack_ref/1)]
      assert outputs == [IR.pack_ref({:instr, 0})]
    end

    test "100-add chain preserves structure and capture count" do
      bias = f32([1.0], [1])
      ir = add_chain_ir(100, bias)

      prog = Program.compile(ir)
      {n_inputs, n_captures, n_consts, opcodes, operands, outputs} = Program.describe(prog)

      assert n_inputs == 1
      assert n_captures == 1
      assert n_consts == 0
      assert length(opcodes) == 100
      assert Enum.all?(opcodes, &(&1 == IR.opcode(:add)))
      # instr 0 adds input0 + capture0; instr k adds instr_{k-1} + capture0.
      assert hd(operands) == Enum.map([{:input, 0}, {:capture, 0}], &IR.pack_ref/1)
      assert Enum.at(operands, 1) == Enum.map([{:instr, 0}, {:capture, 0}], &IR.pack_ref/1)
      assert outputs == [IR.pack_ref({:instr, 99})]
    end

    test "an empty program that returns its input verbatim" do
      ir = %IR{n_inputs: 1, instrs: [], outputs: [{:input, 0}]}

      prog = Program.compile(ir)
      {n_inputs, _nc, _nk, opcodes, operands, outputs} = Program.describe(prog)

      assert n_inputs == 1
      assert opcodes == []
      assert operands == []
      assert outputs == [IR.pack_ref({:input, 0})]
    end
  end

  # Build an IR threading input0 through `n` successive `+ capture0` adds:
  # instr 0 = input0 + capture0; instr k = instr_{k-1} + capture0.
  defp add_chain_ir(n, bias_ref) when n > 0 do
    instrs =
      for k <- 0..(n - 1) do
        left = if k == 0, do: {:input, 0}, else: {:instr, k - 1}
        %{opcode: :add, operands: [left, {:capture, 0}]}
      end

    %IR{n_inputs: 1, captures: [bias_ref], instrs: instrs, outputs: [{:instr, n - 1}]}
  end
end
