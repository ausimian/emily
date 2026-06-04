defmodule Emily.ProgramTest do
  @moduledoc """
  CM0 tests for the program-resource replay engine.

  The oracle is the eager per-op Native path (Emily's Compiler-layer
  oracle is "the backend in non-defn mode"): the compiled single-NIF
  replay must produce **bit-identical** results to a fold of `Native.add`
  on the same inputs. Plus: captures keep weights alive across source GC,
  handles are reusable, malformed IR is rejected cleanly (worker
  survives), async parity holds, and repeated replay leaves no memory
  drift. The dispatch-collapse speedup itself is measured by
  `bench/program_dispatch.exs`.
  """
  use ExUnit.Case, async: true

  import Emily.TensorHelpers

  alias Emily.{IR, Native, Program}

  describe "replay vs eager (bit-identical)" do
    test "single add matches Native.add for f32" do
      a = f32([1.0, 2.0, 3.0, 4.0], [4])
      b = f32([10.0, 20.0, 30.0, 40.0], [4])

      [out] = eval_add(a, b)

      assert to_f32_list(out) == to_f32_list(Native.add(worker(), a, b))
      assert to_f32_list(out) == [11.0, 22.0, 33.0, 44.0]
    end

    test "matches Native.add bit-for-bit across dtypes and broadcasting" do
      cases = [
        # f32 with broadcasting [3] + [1]
        {f32([1.0, 2.0, 3.0], [3]), f32([10.0], [1])},
        # s32 (integer add)
        {s32([7, 8, 9], [3]), s32([1, 2, 3], [3])},
        # f32 with fractional values (catches any rounding drift)
        {f32([1.5, -2.5, 0.125], [3]), f32([0.5, 0.5, 0.875], [3])}
      ]

      for {a, b} <- cases do
        [out] = eval_add(a, b)

        assert Native.to_binary(worker(), out) ==
                 Native.to_binary(worker(), Native.add(worker(), a, b))
      end
    end

    test "100-add chain matches an eager fold of Native.add" do
      input = f32([0.0, 1.0, 2.0], [3])
      bias = f32([1.0, 1.0, 1.0], [3])

      prog = Program.compile(add_chain_ir(100, bias))
      [out] = Program.eval(worker(), prog, [input])

      eager = Enum.reduce(1..100, input, fn _, acc -> Native.add(worker(), acc, bias) end)

      assert to_f32_list(out) == to_f32_list(eager)
      assert to_f32_list(out) == [100.0, 101.0, 102.0]
    end
  end

  describe "resource lifetime" do
    test "captured weight survives GC of the source tensor; handle is reusable" do
      input = f32([0.0, 0.0], [2])

      # Build the program in a closure so the `bias` binding goes out of
      # scope; the Program resource then holds the only ref to it.
      prog =
        (fn ->
           bias = f32([5.0, 7.0], [2])
           Program.compile(add_chain_ir(3, bias))
         end).()

      :erlang.garbage_collect()

      [out1] = Program.eval(worker(), prog, [input])
      [out2] = Program.eval(worker(), prog, [input])

      # 3 successive + [5,7] adds onto [0,0].
      assert to_f32_list(out1) == [15.0, 21.0]
      # Same handle, replayed again, same result.
      assert to_f32_list(out2) == [15.0, 21.0]
    end
  end

  describe "error paths (no crash; worker survives)" do
    test "compile rejects an unknown opcode value" do
      assert_raise ArgumentError, ~r/unknown opcode/, fn ->
        Native.compile_program(
          1,
          [],
          [],
          [999],
          [[IR.pack_ref({:input, 0})]],
          [IR.pack_ref({:instr, 0})]
        )
      end
    end

    test "compile rejects a forward/cyclic instr ref" do
      # instr 0 references its own output {:instr, 0} — not a prior instr.
      assert_raise ArgumentError, ~r/forward or cyclic|prior instruction/, fn ->
        Native.compile_program(
          1,
          [],
          [],
          [IR.opcode(:add)],
          [Enum.map([{:input, 0}, {:instr, 0}], &IR.pack_ref/1)],
          [IR.pack_ref({:instr, 0})]
        )
      end
    end

    test "compile rejects an out-of-range input ref" do
      assert_raise ArgumentError, ~r/out of range/, fn ->
        Native.compile_program(
          1,
          [],
          [],
          [IR.opcode(:add)],
          [Enum.map([{:input, 0}, {:input, 5}], &IR.pack_ref/1)],
          [IR.pack_ref({:instr, 0})]
        )
      end
    end

    test "compile rejects opcode/operand length mismatch" do
      assert_raise ArgumentError, ~r/length mismatch/, fn ->
        Native.compile_program(1, [], [], [IR.opcode(:add)], [], [IR.pack_ref({:input, 0})])
      end
    end

    test "eval rejects the wrong number of inputs; worker still usable after" do
      prog =
        Program.compile(%IR{
          n_inputs: 2,
          instrs: [%{opcode: :add, operands: [{:input, 0}, {:input, 1}]}],
          outputs: [{:instr, 0}]
        })

      a = f32([1.0], [1])

      assert_raise ArgumentError, ~r/expected 2 inputs/, fn ->
        Program.eval(worker(), prog, [a])
      end

      # The worker still serves a correct call afterwards.
      [out] = Program.eval(worker(), prog, [a, f32([2.0], [1])])
      assert to_f32_list(out) == [3.0]
    end
  end

  describe "eval modes" do
    test "async and build modes yield the same values as sync" do
      input = f32([1.0, 2.0, 3.0], [3])
      bias = f32([1.0, 1.0, 1.0], [3])
      prog = Program.compile(add_chain_ir(10, bias))

      [sync_out] = Program.eval(worker(), prog, [input], mode: :sync)
      [async_out] = Program.eval(worker(), prog, [input], mode: :async)
      # :build returns the lazy graph; to_f32_list forces the eval.
      [build_out] = Program.eval(worker(), prog, [input], mode: :build)

      assert to_f32_list(sync_out) == [11.0, 12.0, 13.0]
      assert to_f32_list(async_out) == [11.0, 12.0, 13.0]
      assert to_f32_list(build_out) == [11.0, 12.0, 13.0]
    end

    test "rejects an unknown mode" do
      prog = Program.compile(add_chain_ir(1, f32([1.0], [1])))

      assert_raise ArgumentError, ~r/:mode must be/, fn ->
        Program.eval(worker(), prog, [f32([1.0], [1])], mode: :bogus)
      end
    end
  end

  describe "memory" do
    test "repeated replay does not grow active memory with iteration count" do
      input = f32([1.0, 2.0, 3.0], [3])
      bias = f32([1.0, 1.0, 1.0], [3])
      prog = Program.compile(add_chain_ir(50, bias))

      replay = fn n ->
        for _ <- 1..n do
          [out] = Program.eval(worker(), prog, [input])
          _ = to_f32_list(out)
        end

        :erlang.garbage_collect()
        Native.clear_cache()
        Native.get_active_memory()
      end

      # Warm up so the allocator reaches steady state.
      _ = replay.(50)
      after_100 = replay.(100)
      after_400 = replay.(400)

      # A genuine per-replay leak would make active memory scale with the
      # 4x iteration count; assert it stays flat (small allocator slack).
      assert after_400 - after_100 <= 64 * 1024,
             "active memory grew #{after_400 - after_100} bytes over 4x more replays"
    end
  end

  # --- helpers ---

  defp eval_add(a, b) do
    ir = %IR{
      n_inputs: 2,
      instrs: [%{opcode: :add, operands: [{:input, 0}, {:input, 1}]}],
      outputs: [{:instr, 0}]
    }

    Program.eval(worker(), Program.compile(ir), [a, b])
  end

  defp add_chain_ir(n, bias_ref) when n > 0 do
    instrs =
      for k <- 0..(n - 1) do
        left = if k == 0, do: {:input, 0}, else: {:instr, k - 1}
        %{opcode: :add, operands: [left, {:capture, 0}]}
      end

    %IR{n_inputs: 1, captures: [bias_ref], instrs: instrs, outputs: [{:instr, n - 1}]}
  end
end
