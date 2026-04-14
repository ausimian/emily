defmodule Emily.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    # Nothing on the hot path yet. Future: memory/stats agent, stream pool.
    Supervisor.start_link([], strategy: :one_for_one, name: Emily.Supervisor)
  end
end
