defmodule Emily.Soak.StreamConcurrencyTest do
  @moduledoc """
  Stream-per-process concurrency soak: multiple BEAM processes run
  the same deterministic computation concurrently under sustained
  load, each on its own MLX stream (Metal command queue).

  This is the heavier companion to `backend_concurrency_test.exs`.
  """

  use ExUnit.Case, async: false

  @moduletag :soak

  @workers 4
  @iters_per_worker 50

  defp shared_weights do
    s = Emily.Stream.new(:gpu)

    Emily.Stream.with_stream(s, fn ->
      w = Nx.iota({64, 64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(4096.0)
      b = Nx.iota({64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(64.0)
      {w, b}
    end)
  end

  defp workload(x, {w, b}, stream) do
    Emily.Stream.with_stream(stream, fn ->
      x
      |> Nx.dot(w)
      |> Nx.add(b)
      |> Nx.multiply(x)
      |> Nx.sum(axes: [1])
      |> Nx.to_binary()
    end)
  end

  test "concurrent streams produce correct outputs without crash" do
    {w, b} = shared_weights()

    # Baseline: single-process reference
    baseline_stream = Emily.Stream.new(:gpu)

    x =
      Emily.Stream.with_stream(baseline_stream, fn ->
        Nx.iota({8, 64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(512.0)
      end)

    baseline = workload(x, {w, b}, baseline_stream)

    results =
      1..@workers
      |> Task.async_stream(
        fn _worker_id ->
          stream = Emily.Stream.new(:gpu)

          x =
            Emily.Stream.with_stream(stream, fn ->
              Nx.iota({8, 64}, type: {:f, 32}, backend: Emily.Backend) |> Nx.divide(512.0)
            end)

          for _ <- 1..@iters_per_worker do
            workload(x, {w, b}, stream)
          end
          |> Enum.uniq()
        end,
        max_concurrency: @workers,
        timeout: 60_000
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
