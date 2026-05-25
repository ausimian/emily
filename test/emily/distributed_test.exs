defmodule Emily.DistributedTest do
  use ExUnit.Case, async: true

  alias Emily.Distributed.Launcher

  describe "Launcher.hostfile_entries/2" do
    test "one loopback address per rank, in rank order" do
      assert Launcher.hostfile_entries(2) == [
               ["127.0.0.1:18000"],
               ["127.0.0.1:18001"]
             ]
    end

    test "honours a custom base port" do
      assert Launcher.hostfile_entries(3, 6000) == [
               ["127.0.0.1:6000"],
               ["127.0.0.1:6001"],
               ["127.0.0.1:6002"]
             ]
    end

    test "produces JSON MLX's ring backend accepts" do
      json = :json.encode(Launcher.hostfile_entries(2)) |> IO.iodata_to_binary()
      assert json == ~s([["127.0.0.1:18000"],["127.0.0.1:18001"]])
    end
  end

  describe "dedicated CPU worker (issue #112)" do
    test "collectives run on a CPU worker distinct from the default GPU worker" do
      {:ok, _} = Application.ensure_all_started(:emily)

      assert Process.whereis(Emily.MlxStream.Distributed),
             "expected a supervised Emily.MlxStream.Distributed worker"

      refute Emily.MlxStream.distributed_worker() == Emily.MlxStream.default_worker()
    end

    test "init/1 defaults the group worker to the distributed CPU worker" do
      {:ok, _} = Application.ensure_all_started(:emily)

      # No MLX launch env here, so this is a singleton group (size 1).
      group = Emily.Distributed.init(backend: "ring")
      assert group.worker == Emily.MlxStream.distributed_worker()
    end
  end

  # Multi-rank collectives need the compiled NIF and N OS processes;
  # see Emily.Distributed.Launcher.run/3 for an end-to-end ring all_sum.
end
