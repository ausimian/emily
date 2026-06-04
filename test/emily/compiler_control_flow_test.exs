defmodule Emily.CompilerControlFlowTest do
  @moduledoc """
  CM4 — `Nx.Defn` control-flow constructs compile single-NIF, bit-identical
  to the Evaluator. `:cond` lowers to a `where`-chain (all branches
  evaluated; the value matches the chosen branch exactly).
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  @native [compiler: Emily.Compiler, native: true]
  @eval [compiler: Emily.Compiler]

  defn if_fn(x) do
    if Nx.greater(Nx.sum(x), 0) do
      Nx.multiply(x, 2)
    else
      Nx.negate(x)
    end
  end

  defn cond3_fn(x) do
    s = Nx.sum(x)

    cond do
      Nx.greater(s, 10) -> Nx.multiply(x, 10)
      Nx.greater(s, 0) -> Nx.multiply(x, 2)
      true -> Nx.negate(x)
    end
  end

  defn nested_if_fn(x) do
    y = if Nx.greater(Nx.sum(x), 0), do: Nx.add(x, 1), else: Nx.subtract(x, 1)
    if Nx.greater(Nx.reduce_max(y), 5), do: Nx.divide(y, 2), else: y
  end

  defp equiv(fun, x) do
    native = Nx.Defn.jit(fun, @native).(x)
    eval = Nx.Defn.jit(fun, @eval).(x)
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  describe ":cond / if" do
    test "two-branch if selects the right branch" do
      for data <- [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]] do
        equiv(&if_fn/1, Nx.tensor(data, backend: Emily.Backend))
      end
    end

    test "multi-clause cond picks the first matching predicate" do
      # sums: 12 (>10), 3 (0..10), -6 (else) exercise each clause.
      for data <- [[5.0, 4.0, 3.0], [1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]] do
        equiv(&cond3_fn/1, Nx.tensor(data, backend: Emily.Backend))
      end
    end

    test "nested conds compose" do
      for data <- [[3.0, 3.0, 3.0], [-1.0, 0.0, 1.0], [10.0, 10.0, 10.0]] do
        equiv(&nested_if_fn/1, Nx.tensor(data, backend: Emily.Backend))
      end
    end
  end
end
