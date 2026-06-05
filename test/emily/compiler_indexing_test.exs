defmodule Emily.CompilerIndexingTest do
  @moduledoc """
  CM13 — the gather / stack / cumulative ops Bumblebee's generation loop
  needs, lowered single-NIF and bit-identical to the Evaluator. (The
  generation conformance test exercises them in anger; these pin each op in
  isolation.) Run under `native_fallback: :raise` so a regression raises
  rather than silently falling back.
  """
  use ExUnit.Case, async: true

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  defp t(data, opts \\ []), do: Nx.tensor(data, [backend: Emily.Backend] ++ opts)

  defp equiv(fun, args) do
    native = apply(Nx.Defn.jit(fun, @native), args)
    eval = apply(Nx.Defn.jit(fun, @eval), args)
    assert %Emily.Backend{} = native.data
    assert native.shape == eval.shape and native.type == eval.type
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  test "cumulative sum/product/max/min along the last axis (forward and reverse)" do
    x = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    for op <- [:cumulative_sum, :cumulative_product, :cumulative_max, :cumulative_min] do
      equiv(fn x -> apply(Nx, op, [x, [axis: 1]]) end, [x])
      equiv(fn x -> apply(Nx, op, [x, [axis: 1, reverse: true]]) end, [x])
    end
  end

  test "single-axis gather" do
    x = t([10.0, 20.0, 30.0, 40.0])
    idx = t([[3], [1], [0]], type: :s64)
    out = equiv(fn x, i -> Nx.gather(x, i) end, [x, idx])
    assert Nx.to_flat_list(out) == [40.0, 20.0, 10.0]
  end

  test "multi-axis gather (the sampling path)" do
    x = t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = t([[0, 1], [2, 0]], type: :s64)
    out = equiv(fn x, i -> Nx.gather(x, i) end, [x, idx])
    assert Nx.to_flat_list(out) == [2.0, 5.0]
  end

  test "stack along a new axis" do
    a = t([1.0, 2.0, 3.0])
    b = t([4.0, 5.0, 6.0])
    equiv(fn a, b -> Nx.stack([a, b]) end, [a, b])
    equiv(fn a, b -> Nx.stack([a, b], axis: 1) end, [a, b])
  end
end
