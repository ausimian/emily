defmodule Emily.DistributedTest do
  use ExUnit.Case, async: true

  alias Emily.Distributed.Launcher

  describe "Launcher.hostfile_entries/2" do
    test "one loopback address per rank, in rank order" do
      assert Launcher.hostfile_entries(2) == [
               ["127.0.0.1:5000"],
               ["127.0.0.1:5001"]
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
      assert json == ~s([["127.0.0.1:5000"],["127.0.0.1:5001"]])
    end
  end

  # Multi-rank collectives need the compiled NIF and N OS processes;
  # see Emily.Distributed.Launcher.run/3 for an end-to-end ring all_sum.
end
