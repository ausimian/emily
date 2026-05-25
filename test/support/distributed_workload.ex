defmodule Emily.Test.DistributedWorkload do
  @moduledoc false
  # Rank workload executed *on a launched peer node* by
  # Emily.Distributed.Launcher. It must live in a compiled module (not an
  # inline closure) so the captured `&ring_all_sum/0` reference carries
  # across nodes via the shared `-pa` code path.
  #
  # Each peer is its own BEAM node, so it starts the :emily app itself
  # (Emily.Backend pulls its worker from the Emily.MlxStream GenServer).

  @doc """
  Init the ring group, contribute a `{4}` vector full of this rank, and
  all_sum across ranks. With ranks 0..n-1 the result is sum(0..n-1) in
  every element. Returns `%{rank:, size:, sum:}`.
  """
  def ring_all_sum do
    {:ok, _} = Application.ensure_all_started(:emily)

    group = Emily.Distributed.init(backend: "ring")

    x = Nx.broadcast(Nx.tensor(group.rank * 1.0, backend: Emily.Backend), {4})
    sum = group |> Emily.Distributed.all_sum(x) |> Nx.to_flat_list()

    %{rank: group.rank, size: group.size, sum: sum}
  end
end
