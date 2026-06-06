defmodule Emily.ProgramMemoryTest do
  @moduledoc """
  Memory leak-detection for the program replay engine — split out of
  `Emily.ProgramTest` so it can run `async: false`.

  These assert *tiny* deltas (64 KB / 4 MB) on the **process-global** MLX
  active-memory metric (`Native.get_active_memory/0`), measured across two
  points seconds apart. That metric is shared across the whole VM, so a
  parallel test that loads model weights (the `:conformance` / `*_full`
  lanes in a full `--include` run) inflates the reading mid-measurement and
  the deltas blow up — a false positive, not a real leak. `async: false`
  gives these tests the exclusive run the global metric requires; the rest
  of `Emily.ProgramTest` stays `async: true`.
  """
  use ExUnit.Case, async: false

  import Emily.TensorHelpers

  alias Emily.{IR, Native, Program}

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

  # Local copy of `Emily.ProgramTest`'s IR builder — a chain of `n` adds of
  # `bias_ref` onto input 0. Kept here (rather than shared) so this module
  # stands alone; `Emily.ProgramTest` still uses its own copy.
  defp add_chain_ir(n, bias_ref) when n > 0 do
    instrs =
      for k <- 0..(n - 1) do
        left = if k == 0, do: {:input, 0}, else: {:instr, k - 1}
        %{opcode: :add, operands: [left, {:capture, 0}]}
      end

    %IR{n_inputs: 1, captures: [bias_ref], instrs: instrs, outputs: [{:instr, n - 1}]}
  end
end
