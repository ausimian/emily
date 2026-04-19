defmodule Emily.AsyncEvalTest do
  use ExUnit.Case, async: true

  # Regression tests for the async `Emily.Native.eval/2` path.
  # `eval/2` internally dispatches to `eval_nif/2` (which returns a
  # ref) and awaits the worker's reply via `Emily.Native.Async.call/1`.
  #
  # Higher-level tests in backend_test.exs exercise eval transitively.
  # These tests pin specific properties of the async path:
  # - The caller's mailbox stays empty around an eval.
  # - A process killed mid-eval does not leak, and the worker is
  #   still usable by other callers afterward.
  # - Latency on an already-resident tensor is small (plumbing
  #   overhead, not a pessimisation versus the prior sync path).

  describe "round-trip" do
    test "returns :ok and leaves an empty mailbox" do
      stream = Emily.Stream.new(:gpu)
      worker = stream.worker

      t = Emily.Native.zeros(worker, [4, 4], {:f, 32})
      :ok = Emily.Native.eval(worker, t)

      assert {:message_queue_len, 0} = Process.info(self(), :message_queue_len)
    end

    test "thousands of evals from one process drain cleanly" do
      stream = Emily.Stream.new(:gpu)
      worker = stream.worker

      t = Emily.Native.zeros(worker, [4, 4], {:f, 32})

      for _ <- 1..1_000 do
        :ok = Emily.Native.eval(worker, t)
      end

      assert {:message_queue_len, 0} = Process.info(self(), :message_queue_len)
    end
  end

  describe "robustness" do
    test "caller killed mid-eval; worker remains usable" do
      stream = Emily.Stream.new(:gpu)
      worker = stream.worker

      # A doomed process fires off many evals then exits. The worker
      # will deliver replies to a dead PID; enif_send silently drops
      # them. The worker must still process subsequent work from
      # live callers.
      {pid, mon} =
        spawn_monitor(fn ->
          t = Emily.Native.zeros(worker, [64, 64], {:f, 32})

          for _ <- 1..100 do
            _ref = Emily.Native.eval_nif(worker, t)
          end

          exit(:normal)
        end)

      receive do
        {:DOWN, ^mon, :process, ^pid, _} -> :ok
      after
        5_000 -> flunk("doomed process did not exit")
      end

      # Let the worker drain any queued evals from the dead caller.
      Process.sleep(100)

      # The worker should still respond to fresh evals.
      t = Emily.Native.ones(worker, [4, 4], {:f, 32})
      :ok = Emily.Native.eval(worker, t)
    end
  end

  describe "latency" do
    test "eval of a resident tensor completes in under 1 ms on average" do
      stream = Emily.Stream.new(:gpu)
      worker = stream.worker

      t = Emily.Native.zeros(worker, [4, 4], {:f, 32})
      :ok = Emily.Native.eval(worker, t)

      n = 1_000

      {us, :ok} =
        :timer.tc(fn ->
          for _ <- 1..n, do: :ok = Emily.Native.eval(worker, t)
          :ok
        end)

      per_op_us = us / n

      # Budget: 1 ms per eval on a cached 4×4 tensor is very
      # generous — observed locally is ~5-20 µs. A regression that
      # pushes this over 1000 µs indicates something is queuing or
      # scheduling incorrectly.
      assert per_op_us < 1_000,
             "eval averaged #{Float.round(per_op_us, 2)} µs per call"
    end
  end
end
