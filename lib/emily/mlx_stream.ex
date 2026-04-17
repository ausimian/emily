defmodule Emily.MlxStream do
  @moduledoc false
  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))
  end

  def worker(server), do: GenServer.call(server, :worker)

  @doc false
  def default_worker do
    case Process.get(:emily_worker) do
      nil ->
        w = worker(__MODULE__.Default)
        Process.put(:emily_worker, w)
        w

      w ->
        w
    end
  end

  @impl true
  def init(_opts) do
    worker = Emily.Native.create_worker()
    {:ok, %{worker: worker}}
  end

  @impl true
  def handle_call(:worker, _from, %{worker: w} = state) do
    {:reply, w, state}
  end
end
