defmodule Emily.Test.DistributedWorkload do
  @moduledoc false
  # Rank workloads executed *on launched peer nodes* by
  # Emily.Distributed.Launcher. They must live in a compiled module (not
  # inline closures) so the captured `&fun/0` references carry across
  # nodes via the shared `-pa` code path.
  #
  # Each peer is its own BEAM node, so it starts the :emily app itself
  # (Emily.Backend pulls its worker from the Emily.MlxStream GenServer).
  #
  # SPMD discipline: every rank runs the same sequence of collectives in
  # the same order — each collective is a barrier all ranks must reach.

  alias Emily.Distributed

  @vec 3

  @doc """
  Init the ring group, contribute a `{#{@vec}}` vector full of this
  rank, and all_sum across ranks. Result is sum(0..n-1) in every
  element. Returns `%{rank:, size:, sum:}`.
  """
  def ring_all_sum do
    group = init()
    x = filled(group.rank, @vec)
    %{rank: group.rank, size: group.size, sum: flat(Distributed.all_sum(group, x))}
  end

  @doc """
  Run every ring-supported all-to-all collective once, in a fixed
  order, on one group. Each rank contributes known inputs so results
  are exactly predictable:

    * all_sum/all_max/all_min — a `{#{@vec}}` vector full of this rank
    * all_gather — a `{1}` vector `[rank]`, gathered in rank order

  Returns a map of the flattened result of each. (`sum_scatter` is
  omitted: MLX's ring backend raises `[ReduceScatter] Not implemented
  yet`; the binding still exposes it for backends that support it.)
  """
  def all_collectives do
    group = init()
    r = group.rank
    n = group.size

    scalar = filled(r, @vec)
    mine = Nx.tensor([r * 1.0], backend: Emily.Backend)

    %{
      rank: r,
      size: n,
      all_sum: flat(Distributed.all_sum(group, scalar)),
      all_max: flat(Distributed.all_max(group, scalar)),
      all_min: flat(Distributed.all_min(group, scalar)),
      all_gather: flat(Distributed.all_gather(group, mine))
    }
  end

  @doc """
  Point-to-point: rank 0 sends `[10, 20, 30]` to rank 1, which receives
  it. Any other rank stays idle. Returns `%{rank:, role:, value:}`.
  """
  def send_recv do
    group = init()

    case group.rank do
      0 ->
        x = Nx.tensor([10.0, 20.0, 30.0], backend: Emily.Backend)
        # Force eval so the send primitive actually transmits.
        _ = flat(Distributed.send(group, x, 1))
        %{rank: 0, role: :send, value: flat(x)}

      1 ->
        got = Distributed.recv(group, {3}, {:f, 32}, 0)
        %{rank: 1, role: :recv, value: flat(got)}

      r ->
        %{rank: r, role: :idle, value: []}
    end
  end

  defp init do
    {:ok, _} = Application.ensure_all_started(:emily)
    Distributed.init(backend: "ring")
  end

  defp filled(value, size) do
    Nx.broadcast(Nx.tensor(value * 1.0, backend: Emily.Backend), {size})
  end

  defp flat(tensor), do: Nx.to_flat_list(tensor)
end
