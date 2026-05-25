defmodule Emily.DistributedRingTest do
  # Real multi-process collectives over the ring backend: launches local
  # BEAM peer ranks that talk over loopback TCP. Excluded by default
  # (spawns OS processes + binds ports + uses a GPU context per rank);
  # run explicitly:
  #
  #     mix test --only distributed_ring
  use ExUnit.Case

  @moduletag :distributed_ring
  # Each rank is a full BEAM + MLX process; the ring handshake + per-rank
  # init can run past the 60s default on a busy machine.
  @moduletag timeout: 180_000

  alias Emily.Distributed.Launcher
  alias Emily.Test.DistributedWorkload

  test "all_sum reduces across two local ranks" do
    results = Launcher.run(2, &DistributedWorkload.ring_all_sum/0)

    assert ranks(results) == [0, 1]

    # Each rank r contributes a vector of r; the sum across ranks 0 and 1
    # is 1.0 in every element, seen identically by both.
    for r <- results do
      assert r.size == 2
      assert r.sum == [1.0, 1.0, 1.0]
    end
  end

  test "all collectives reduce correctly across four ranks" do
    n = 4
    results = Launcher.run(n, &DistributedWorkload.all_collectives/0, base_port: 18_100)

    assert ranks(results) == Enum.to_list(0..(n - 1))

    sum = n * (n - 1) / 2

    for res <- results do
      assert res.size == n
      assert res.all_sum == List.duplicate(sum, 3)
      assert res.all_max == List.duplicate((n - 1) * 1.0, 3)
      assert res.all_min == List.duplicate(0.0, 3)

      # Gather concatenates each rank's [rank] in rank order.
      assert res.all_gather == Enum.map(0..(n - 1), &(&1 * 1.0))
    end
  end

  test "send/recv passes a tensor from rank 0 to rank 1" do
    results = Launcher.run(2, &DistributedWorkload.send_recv/0, base_port: 18_200)
    by_rank = Map.new(results, &{&1.rank, &1})

    assert by_rank[0].role == :send
    assert by_rank[1].role == :recv
    assert by_rank[1].value == [10.0, 20.0, 30.0]
  end

  defp ranks(results), do: results |> Enum.map(& &1.rank) |> Enum.sort()
end
