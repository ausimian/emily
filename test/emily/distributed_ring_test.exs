defmodule Emily.DistributedRingTest do
  # Real multi-process collective over the ring backend: launches local
  # BEAM peer ranks that talk over loopback TCP. Excluded by default
  # (spawns OS processes + binds ports + uses a GPU context per rank);
  # run explicitly:
  #
  #     mix test --only distributed_ring
  use ExUnit.Case

  @moduletag :distributed_ring

  alias Emily.Distributed.Launcher
  alias Emily.Test.DistributedWorkload

  test "all_sum reduces across two local ranks over the ring backend" do
    results = Launcher.run(2, &DistributedWorkload.ring_all_sum/0)

    assert length(results) == 2

    # Each rank r contributes a vector of r; the all-reduce sum across
    # ranks 0 and 1 is 1.0 in every element, seen identically by both.
    assert Enum.map(results, & &1.rank) |> Enum.sort() == [0, 1]

    for r <- results do
      assert r.size == 2
      assert r.sum == [1.0, 1.0, 1.0, 1.0]
    end
  end
end
