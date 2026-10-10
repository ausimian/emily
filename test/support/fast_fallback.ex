defmodule Emily.FastFallback do
  @moduledoc """
  Runs an `Emily.Fast` call through its composed-defn fallback, as the
  oracle for fused-vs-fallback tests.

  The oracle has to stay on `Nx.BinaryBackend` from start to finish.
  `Nx.Defn.Evaluator` dispatches each `Nx.block` to the backend of its
  arguments, so an argument on `Emily.Backend` sends the block to the
  fused kernel instead. A "reference" built as
  `Nx.iota(..., backend: Nx.BinaryBackend) |> Nx.divide(100)` is
  already on `Emily.Backend` when that is the default, because the
  scalar is created on the default backend. Constants created inside
  the fallback move it the same way. So `run/2` copies the arguments
  to `Nx.BinaryBackend`, makes it the default for the call, and checks
  that the result stayed there.
  """

  import ExUnit.Assertions

  @doc """
  Applies `fun` to `args` under `Nx.Defn.Evaluator` on
  `Nx.BinaryBackend` and returns the result.
  """
  def run(fun, args) when is_function(fun) and is_list(args) do
    args = Enum.map(args, &Nx.backend_copy(&1, Nx.BinaryBackend))

    result =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
      end)

    assert %Nx.Tensor{data: %Nx.BinaryBackend{}} = result,
           "fallback oracle left Nx.BinaryBackend: #{inspect(result.data.__struct__)}"

    result
  end
end
