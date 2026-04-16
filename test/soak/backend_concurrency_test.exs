defmodule Emily.Soak.ConcurrencyTest do
  @moduledoc """
  Cross-process determinism + safety check for the Backend layer:
  run a deterministic Nx computation in separate BEAM processes and
  assert every process produces bit-identical output.

  Each worker uses its own `Emily.Stream` so concurrent dispatch
  targets separate Metal command queues. Without per-process streams,
  concurrent dispatch on the shared default stream trips Metal
  assertions like `A command encoder is already encoding to this
  command buffer` — see `stream_concurrency_test.exs` for the
  heavier concurrency soak.
  """

  use ExUnit.Case, async: false

  @moduletag :soak

  @workers 8
  @iters_per_worker 10

  defp fresh_input(s) do
    Emily.Stream.with_stream(s, fn ->
      Nx.iota({8, 64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(512.0)
    end)
  end

  defp workload(x, s) do
    Emily.Stream.with_stream(s, fn ->
      x
      |> Nx.multiply(x)
      |> Nx.add(Nx.tensor(1.0, backend: Emily.Backend))
      |> Nx.sum(axes: [1])
      |> Nx.to_binary()
    end)
  end

  test "parallel workers produce bit-identical outputs" do
    # Baseline: single-process reference on its own stream.
    baseline_stream = Emily.Stream.new(:gpu)
    baseline = workload(fresh_input(baseline_stream), baseline_stream)

    results =
      1..@workers
      |> Task.async_stream(
        fn _ ->
          stream = Emily.Stream.new(:gpu)
          x = fresh_input(stream)

          for _ <- 1..@iters_per_worker do
            workload(x, stream)
          end
          |> Enum.uniq()
        end,
        max_concurrency: @workers,
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
