defmodule Emily.CompilerSelectionTest do
  @moduledoc """
  CM8 — selection / sort ops compile single-NIF, bit-identical to the
  Evaluator-on-`Emily.Backend` path: `argmax`, `argmin`, `clip`, `sort`,
  `argsort` (both directions). Each lowering mirrors the matching
  `Emily.Backend` callback, so the compiled path can't drift from eager.

  Run under `native_fallback: :raise` so an op that fails to lower raises
  rather than silently passing via the evaluator — these are coverage gates.
  """
  use ExUnit.Case, async: true

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  defp t(data, opts \\ []), do: Nx.tensor(data, [backend: Emily.Backend] ++ opts)

  # Assert the native single-NIF path matches the evaluator bit-for-bit.
  defp equiv(fun, x) do
    native = Nx.Defn.jit(fun, @native).(x)
    eval = Nx.Defn.jit(fun, @eval).(x)
    assert %Emily.Backend{} = native.data
    assert native.shape == eval.shape
    assert native.type == eval.type
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  describe "argmax / argmin" do
    setup do
      %{x: t([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])}
    end

    test "argmax over each axis", %{x: x} do
      equiv(&Nx.argmax(&1, axis: 0), x)
      equiv(&Nx.argmax(&1, axis: 1), x)
    end

    test "argmin over each axis", %{x: x} do
      equiv(&Nx.argmin(&1, axis: 0), x)
      equiv(&Nx.argmin(&1, axis: 1), x)
    end

    test "keep_axis retains the reduced dimension", %{x: x} do
      out = equiv(&Nx.argmax(&1, axis: 1, keep_axis: true), x)
      assert out.shape == {2, 1}
    end
  end

  describe "clip" do
    test "clamps to [min, max]" do
      x = t([[-1.0, 0.5, 3.0], [4.0, 9.0, -2.0]])
      equiv(&Nx.clip(&1, 0.0, 4.0), x)
    end

    test "integer clamp" do
      x = t([-5, 0, 7, 12], type: :s32)
      equiv(&Nx.clip(&1, 0, 10), x)
    end
  end

  describe "sort / argsort" do
    setup do
      %{x: t([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])}
    end

    test "sort ascending and descending along an axis", %{x: x} do
      equiv(&Nx.sort(&1, axis: 1, direction: :asc), x)
      equiv(&Nx.sort(&1, axis: 1, direction: :desc), x)
      equiv(&Nx.sort(&1, axis: 0, direction: :desc), x)
    end

    test "argsort ascending and descending along an axis", %{x: x} do
      equiv(&Nx.argsort(&1, axis: 1, direction: :asc), x)
      equiv(&Nx.argsort(&1, axis: 1, direction: :desc), x)
      equiv(&Nx.argsort(&1, axis: 0, direction: :desc), x)
    end
  end
end
