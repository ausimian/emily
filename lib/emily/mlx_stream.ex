defmodule Emily.MlxStream do
  @moduledoc false
  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))
  end

  def worker(server), do: GenServer.call(server, :worker)

  @doc false
  def default_worker do
    cached_worker(:emily_worker, __MODULE__.Default)
  end

  @doc false
  # CPU worker reserved for distributed collectives. Kept separate from
  # the default GPU worker so a blocking collective eval (ring handshake,
  # all-reduce waiting on peers) never stalls ordinary GPU inference.
  def distributed_worker do
    cached_worker(:emily_distributed_worker, __MODULE__.Distributed)
  end

  defp cached_worker(key, server) do
    case Process.get(key) do
      nil ->
        w = worker(server)
        Process.put(key, w)
        w

      w ->
        w
    end
  end

  @impl true
  def init(opts) do
    worker =
      case Keyword.get(opts, :device, :gpu) do
        :cpu -> Emily.Native.create_cpu_worker()
        :gpu -> Emily.Native.create_worker()
      end

    {:ok, %{worker: worker}}
  end

  @impl true
  def handle_call(:worker, _from, %{worker: w} = state) do
    {:reply, w, state}
  end
end
