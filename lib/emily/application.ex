defmodule Emily.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Emily.MlxStream, name: Emily.MlxStream.Default}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Emily.Supervisor)
  end
end
