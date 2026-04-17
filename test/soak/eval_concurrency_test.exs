defmodule Emily.Soak.EvalConcurrencyTest do
  @moduledoc """
  Stress test for concurrent mx::eval without the safe_eval mutex.

  Validates that MLX's thread-local CommandEncoder (ml-explore/mlx#3348)
  prevents the SIGABRT/SIGSEGV that previously occurred when multiple
  BEAM dirty-CPU scheduler threads called mx::eval simultaneously.

  Deliberately uses the **default stream** (no per-process streams) to
  exercise the exact crash vector the mutex was protecting against:
  concurrent `to_binary` -> `mx::eval` from multiple dirty-CPU threads
  on shared tensors.
  """

  use ExUnit.Case, async: false

  @moduletag :soak

  @workers 16
  @iters_per_worker 100

  test "concurrent to_binary on shared tensors without crash" do
    shared = Nx.iota({32, 128}, type: {:f, 32}, backend: Emily.Backend)
    baseline = Nx.to_binary(shared)

    results =
      1..@workers
      |> Task.async_stream(
        fn _ ->
          for _ <- 1..@iters_per_worker do
            Nx.to_binary(shared)
          end
          |> Enum.uniq()
        end,
        max_concurrency: @workers,
        timeout: 120_000
      )
      |> Enum.map(fn {:ok, v} -> v end)

    for {result, i} <- Enum.with_index(results) do
      assert length(result) == 1,
             "worker #{i} produced non-deterministic output (#{length(result)} distinct)"

      [only] = result

      assert only == baseline,
             "worker #{i} output differs from baseline"
    end
  end
end
