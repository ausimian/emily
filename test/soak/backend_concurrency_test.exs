defmodule Emily.Soak.ConcurrencyTest do
  @moduledoc """
  Cross-process determinism check for the Backend layer: run a
  deterministic Nx computation in separate BEAM processes and assert
  every process produces bit-identical output.

  ## Why this test runs with `max_concurrency: 1`

  MLX's Metal backend is not safe for concurrent kernel dispatch from
  multiple OS threads. Running several BEAM processes that hit the
  dirty-CPU scheduler simultaneously trips Metal assertions like
  `A command encoder is already encoding to this command buffer`.
  That is a library-wide constraint inherited from Apple's MLX runtime
  — not an Emily-side race. The intended deployment model is a single
  consumer process fed by `Nx.Serving`, which already serialises work.

  So this test exercises the **determinism** half of the PLAN
  requirement ("parallel inference; determinism + no crashes"). True
  multi-threaded kernel dispatch is blocked upstream until MLX adds
  stream-per-thread support; we'll revisit then.
  """

  use ExUnit.Case, async: false

  @moduletag :soak

  @workers 8
  @iters_per_worker 10

  defp fresh_input do
    Nx.iota({8, 64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(512.0)
  end

  defp workload(x) do
    x
    |> Nx.multiply(x)
    |> Nx.add(Nx.tensor(1.0, backend: Emily.Backend))
    |> Nx.sum(axes: [1])
    |> Nx.to_binary()
  end

  test "parallel workers produce bit-identical outputs" do
    # Baseline: single-process reference using a fresh tensor.
    baseline = workload(fresh_input())

    results =
      1..@workers
      |> Task.async_stream(
        fn _ ->
          x = fresh_input()

          for _ <- 1..@iters_per_worker do
            workload(x)
          end
          |> Enum.uniq()
        end,
        # Serialised by design — see @moduledoc.
        max_concurrency: 1,
        timeout: 30_000
      )
      |> Enum.map(fn {:ok, v} -> v end)

    # Every worker should have seen the same deterministic output every
    # iteration, so each list should be a single-element list.
    for {result, i} <- Enum.with_index(results) do
      assert length(result) == 1,
             "worker #{i} produced non-deterministic output (#{length(result)} distinct)"

      [only] = result

      assert only == baseline,
             "worker #{i} output differs from baseline"
    end
  end
end
