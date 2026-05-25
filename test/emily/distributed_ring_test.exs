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

  test "a pending collective does not block the shared GPU worker" do
    {:ok, %{peers: peers, hostfile: hostfile}} = Launcher.start(2, base_port: 18_300)
    [{0, p0, _}, {1, p1, _}] = peers

    # Rank 1 joins the ring and idles forever (it never issues all_sum),
    # so rank 0's collective stays pending. We never await this call;
    # it's torn down with the peer.
    idle =
      Task.async(fn ->
        :peer.call(p1, DistributedWorkload, :gpu_free_while_collective_pending, [], :infinity)
      end)

    try do
      # With the collective hanging, rank 0's ordinary GPU op must still
      # complete. A bug that puts the collective's eval on the shared GPU
      # worker would stall this call until the :peer.call timeout.
      r0 = :peer.call(p0, DistributedWorkload, :gpu_free_while_collective_pending, [], 60_000)

      assert r0.rank == 0
      assert r0.collective_pending, "collective should still be in flight"
      assert r0.gpu_result == [2.0, 4.0, 6.0]
    after
      Task.shutdown(idle, :brutal_kill)
      Launcher.stop(peers)
      File.rm(hostfile)
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
