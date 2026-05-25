defmodule Emily.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    Emily.Telemetry.init_dedup_table()

    children = [
      {Emily.MlxStream, name: Emily.MlxStream.Default},
      # Dedicated CPU worker for distributed collectives — keeps their
      # blocking eval off the shared GPU worker (see Emily.Distributed).
      Supervisor.child_spec(
        {Emily.MlxStream, name: Emily.MlxStream.Distributed, device: :cpu},
        id: Emily.MlxStream.Distributed
      )
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Emily.Supervisor)
  end
end
