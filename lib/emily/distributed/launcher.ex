defmodule Emily.Distributed.Launcher do
  @moduledoc """
  Launch N ranks as BEAM peer nodes for MLX's `ring` backend.

  This is the `:peer`-based equivalent of `mlx.launch -n N --backend
  ring`, but it yields supervised BEAM nodes instead of fire-and-forget
  OS processes. Each peer is one rank; `start/2` writes a loopback
  hostfile and starts the peers with `MLX_RANK`/`MLX_HOSTFILE` in their
  environment, after which each can call `Emily.Distributed.init/1`.

  ## Single node now, multiple Macs later

  As written this launches **local** peers over loopback — enough to
  develop and test collectives on one machine (the `ring` backend
  connects ranks over `127.0.0.1`). The same shape extends to multiple
  Macs by giving each peer an ssh `exec` and a `standard_io` connection:

      :peer.start_link(%{
        exec: {~c"/usr/bin/ssh", [host, ~c"erl"]},
        connection: :standard_io,
        env: [...]
      })

  and supplying the real Thunderbolt-bridge IPs in the hostfile instead
  of loopback. Control flows over the ssh stdio pipe; the tensor data
  plane is MLX's own transport.

  ## Caveat

  Each peer must be able to load the emily NIF — it inherits this node's
  code paths via `-pa`, so it works from a normal `mix`/release context
  on the same machine. The connection is `:standard_io`, so no Erlang
  distribution (cookie/EPMD) is required between ranks.

  ## Example

      Emily.Distributed.Launcher.run(2, fn ->
        g = Emily.Distributed.init(backend: "ring")
        x = Nx.broadcast(Nx.tensor(Emily.Distributed.rank(g)), {4})
        Emily.Distributed.all_sum(g, x) |> Nx.to_flat_list()
      end)
      # => [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]   (per rank)
  """

  @type peer :: {rank :: non_neg_integer(), pid :: pid(), node :: node()}

  @doc """
  Build the ring hostfile entries: one `"127.0.0.1:port"` per rank, in
  rank order. Pure — useful to inspect/test without launching anything.
  """
  @spec hostfile_entries(pos_integer(), pos_integer()) :: [[String.t()]]
  def hostfile_entries(n, base_port \\ 5000) when n >= 1 do
    for rank <- 0..(n - 1), do: ["127.0.0.1:#{base_port + rank}"]
  end

  @doc """
  Start `n` local peer ranks. Returns `{:ok, %{peers: [peer], hostfile:
  path}}`. Caller is responsible for `stop/1`.

  ## Options

    * `:base_port` — first TCP port for the ring (default `5000`).
    * `:verbose` — set `MLX_RING_VERBOSE` on the peers (default `false`).
  """
  @spec start(pos_integer(), keyword()) ::
          {:ok, %{peers: [peer()], hostfile: Path.t()}}
  def start(n, opts \\ []) when n >= 2 do
    base_port = Keyword.get(opts, :base_port, 5000)
    verbose = if Keyword.get(opts, :verbose, false), do: ~c"1", else: ~c"0"

    hostfile = write_hostfile(n, base_port)
    code_args = Enum.flat_map(:code.get_path(), &[~c"-pa", &1])

    peers =
      for rank <- 0..(n - 1) do
        {:ok, pid, node} =
          :peer.start_link(%{
            name: :"emily_rank#{rank}",
            connection: :standard_io,
            args: code_args,
            env: [
              {~c"MLX_RANK", ~c"#{rank}"},
              {~c"MLX_HOSTFILE", String.to_charlist(hostfile)},
              {~c"MLX_RING_VERBOSE", verbose}
            ]
          })

        {rank, pid, node}
      end

    {:ok, %{peers: peers, hostfile: hostfile}}
  end

  @doc """
  Run a zero-arity `fun` on every rank concurrently and return the
  per-rank results in rank order, then tear the peers down.

  The function runs on each peer node, so it must be a captured
  module-function (anonymous closures don't carry across nodes cleanly).
  Ranks run concurrently because `Emily.Distributed.init/1` blocks until
  every peer has joined the ring.
  """
  @spec run(pos_integer(), (-> term()), keyword()) :: [term()]
  def run(n, fun, opts \\ []) when is_function(fun, 0) do
    {:ok, %{peers: peers, hostfile: hostfile}} = start(n, opts)

    try do
      peers
      |> Enum.map(fn {_rank, pid, _node} ->
        Task.async(fn -> :peer.call(pid, :erlang, :apply, [fun, []], :infinity) end)
      end)
      |> Task.await_many(:infinity)
    after
      stop(peers)
      File.rm(hostfile)
    end
  end

  @doc "Stop all peer nodes started by `start/2`."
  @spec stop([peer()]) :: :ok
  def stop(peers) do
    Enum.each(peers, fn {_rank, pid, _node} -> :peer.stop(pid) end)
  end

  defp write_hostfile(n, base_port) do
    path =
      Path.join(
        System.tmp_dir!(),
        "emily_ring_#{System.unique_integer([:positive])}.json"
      )

    File.write!(path, :json.encode(hostfile_entries(n, base_port)))
    path
  end
end
