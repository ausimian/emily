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
  alias Nx.Defn.{Composite, Expr}

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

  describe "attribute-carrying ops (replay vs eager)" do
    test "unary + cast + binary chain matches eager" do
      # f(x, y) = tanh(x) * astype(y, f32)
      x = f32([0.1, 0.2, -0.3, 0.5], [4])
      y = f32([1.0, 2.0, 3.0, 4.0], [4])

      ir = %IR{
        n_inputs: 2,
        instrs: [
          %{opcode: :tanh, operands: [{:input, 0}]},
          %{opcode: :astype, operands: [{:input, 1}], iattrs: [[IR.dtype_code({:f, 32})]]},
          %{opcode: :multiply, operands: [{:instr, 0}, {:instr, 1}]}
        ],
        outputs: [{:instr, 2}]
      }

      [out] = Program.eval(worker(), Program.compile(ir), [x, y])

      eager =
        Native.multiply(
          worker(),
          Native.tanh(worker(), x),
          Native.astype(worker(), y, {:f, 32})
        )

      assert to_f32_list(out) == to_f32_list(eager)
    end

    test "reshape + transpose match eager" do
      x = f32(Enum.map(1..6, &(&1 * 1.0)), [6])

      ir = %IR{
        n_inputs: 1,
        instrs: [
          %{opcode: :reshape, operands: [{:input, 0}], iattrs: [[2, 3]]},
          %{opcode: :transpose, operands: [{:instr, 0}], iattrs: [[1, 0]]}
        ],
        outputs: [{:instr, 1}]
      }

      [out] = Program.eval(worker(), Program.compile(ir), [x])
      eager = Native.transpose(worker(), Native.reshape(worker(), x, [2, 3]), [1, 0])

      assert Native.shape(out) == [3, 2]
      assert to_f32_list(out) == to_f32_list(eager)
    end

    test "broadcast_to matches eager" do
      x = f32([1.0, 2.0, 3.0], [3])

      ir = %IR{
        n_inputs: 1,
        instrs: [%{opcode: :broadcast_to, operands: [{:input, 0}], iattrs: [[2, 3]]}],
        outputs: [{:instr, 0}]
      }

      [out] = Program.eval(worker(), Program.compile(ir), [x])

      assert Native.shape(out) == [2, 3]
      assert to_f32_list(out) == to_f32_list(Native.broadcast_to(worker(), x, [2, 3]))
    end

    test "describe round-trips iattrs" do
      ir = %IR{
        n_inputs: 1,
        instrs: [%{opcode: :reshape, operands: [{:input, 0}], iattrs: [[2, 3]]}],
        outputs: [{:instr, 0}]
      }

      {_n, _nc, _nk, opcodes, _operands, iattrs, _outputs} =
        Program.describe(Program.compile(ir))

      assert opcodes == [IR.opcode(:reshape)]
      assert iattrs == [[[2, 3]]]
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
          [[]],
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
          [[]],
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
          [[]],
          [IR.pack_ref({:instr, 0})]
        )
      end
    end

    test "compile rejects opcode/operand length mismatch" do
      assert_raise ArgumentError, ~r/length mismatch/, fn ->
        Native.compile_program(1, [], [], [IR.opcode(:add)], [], [[]], [IR.pack_ref({:input, 0})])
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

    test "compiled mode (mx::compile) matches sync, bit-for-bit, and caches" do
      input = f32([1.0, 2.0, 3.0], [3])
      bias = f32([1.0, 1.0, 1.0], [3])
      prog = Program.compile(add_chain_ir(20, bias))

      [sync_out] = Program.eval(worker(), prog, [input], mode: :sync)
      # First compiled eval builds + caches the mx::compile'd replay;
      # subsequent ones hit the cache. All must equal the sync result.
      [c1] = Program.eval(worker(), prog, [input], mode: :compiled)
      [c2] = Program.eval(worker(), prog, [input], mode: :compiled)
      [c3] = Program.eval(worker(), prog, [f32([4.0, 5.0, 6.0], [3])], mode: :compiled)

      assert to_f32_list(c1) == to_f32_list(sync_out)
      assert to_f32_list(c2) == to_f32_list(sync_out)
      # Same compiled program, fresh inputs (shape-stable): 6/7/8 + 20.
      assert to_f32_list(c3) == [24.0, 25.0, 26.0]
    end

    test "compiled mode on a matmul + softmax block matches sync within f32 tol" do
      # mx::compile leaves the matmuls alone but fuses the elementwise
      # softmax run (max/sub/exp/sum/div). On a deep graph that fusion
      # reassociates f32 arithmetic, so the compiled result matches sync
      # to within a few ULP rather than bit-for-bit -- assert a tight
      # absolute tolerance. (The add-chain above stays bit-identical
      # because it is shallow and reassociation-free.)
      d = 8
      seq = 4
      shapes = [{seq, d}, {d, d}]
      [a_ref, b_ref] = Enum.map(shapes, &rand_f32/1)

      fun = fn [a, b] ->
        s = Nx.dot(a, b)
        m = Nx.reduce_max(s, axes: [-1], keep_axes: true)
        e = Nx.exp(Nx.subtract(s, m))
        w = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
        Nx.dot(w, b)
      end

      prog = trace_to_program(fun, shapes)
      [sync_out] = Program.eval(worker(), prog, [a_ref, b_ref], mode: :sync)
      [comp_out] = Program.eval(worker(), prog, [a_ref, b_ref], mode: :compiled)

      drift =
        Enum.zip(to_f32_list(sync_out), to_f32_list(comp_out))
        |> Enum.reduce(0.0, fn {x, y}, acc -> max(acc, abs(x - y)) end)

      assert Native.shape(comp_out) == [seq, d]
      assert drift <= 1.0e-5, "compiled vs sync drift #{drift} exceeds 1.0e-5"
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

    test "compiled-mode programs release their mx::compile cache on GC" do
      # Each distinct Program evaled in :compiled mode installs an entry in
      # the worker's *thread-local* mx::compile cache that pins copies of
      # its captured weights. Program::~Program must drop that entry on the
      # worker thread; if it instead erased the GC thread's cache (the bug
      # this guards), the weights would stay live and active memory would
      # scale with the number of compiled programs.
      n = div(512 * 1024, 4)
      zero = f32(List.duplicate(0.0, n), [n])

      make_and_run = fn k ->
        # A fresh capture per program -> a distinct cache entry / fun_id.
        weight = f32(List.duplicate(k * 1.0, n), [n])
        prog = Program.compile(add_chain_ir(4, weight))
        [out] = Program.eval(worker(), prog, [zero], mode: :compiled)
        _ = to_f32_list(out)
        :ok
      end

      flush = fn ->
        :erlang.garbage_collect()
        # Teardown is posted to the worker queue during resource GC; a sync
        # op after it (FIFO) guarantees every posted teardown has run before
        # we read memory.
        [out] = Program.eval(worker(), Program.compile(add_chain_ir(1, zero)), [zero])
        _ = to_f32_list(out)
        Native.clear_cache()
        Native.get_active_memory()
      end

      run_n = fn count ->
        for k <- 1..count, do: make_and_run.(k)
        flush.()
      end

      _ = run_n.(20)
      after_40 = run_n.(40)
      after_80 = run_n.(80)

      # A leaked cache entry pins ~512 KiB per program; 2x more programs
      # would add tens of MiB. Assert it stays flat (generous slack).
      assert after_80 - after_40 <= 4 * 1024 * 1024,
             "compiled-mode cache leaked #{after_80 - after_40} bytes over 2x more programs"
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

  # Trace a function of a parameter list into a `Program` (mirrors what
  # `Emily.Compiler`'s native path does), so a CM6 test can drive the
  # same Program through different eval modes.
  defp trace_to_program(fun, shapes) do
    vars =
      shapes
      |> Enum.with_index()
      |> Enum.map(fn {shape, i} ->
        Expr.parameter(Nx.template(shape, {:f, 32}), :root, i)
      end)

    expr = fun.(vars)

    {_template, leaves_rev} =
      Composite.traverse(expr, [], fn leaf, acc -> {leaf, [leaf | acc]} end)

    leaves_rev |> Enum.reverse() |> IR.lower() |> Program.compile()
  end

  defp rand_f32(shape) do
    dims = Tuple.to_list(shape)
    n = Enum.product(dims)
    bin = for i <- 1..n, into: <<>>, do: <<:math.sin(i * 0.31) * 0.5::float-32-native>>
    Native.from_binary(bin, dims, {:f, 32})
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
