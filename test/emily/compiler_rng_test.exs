defmodule Emily.CompilerRngTest do
  @moduledoc """
  CM12 — `Nx.Random` lowers under the native compiler. threefry2x32 is pure
  tensor ops (bitwise/shift/reshape/concat) plus a `while` whose body indexes
  a rotation table by the loop counter — a *dynamic* (runtime-start) slice —
  and `uniform`/`gumbel` turn random bits into floats via `bitcast`. With
  those wired (CM12), the whole RNG surface compiles, and a PRNG key threads
  through a generation-style loop as ordinary carried state.

  Determinism makes the evaluator the oracle: for a fixed seed the native
  single-NIF path must produce bit-identical draws. Run under
  `native_fallback: :raise` so any un-lowered op fails rather than passing.
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  defp key(seed \\ 7), do: Nx.Random.key(seed) |> Nx.backend_copy(Emily.Backend)

  defp equiv(fun, args) do
    native = apply(Nx.Defn.jit(fun, @native), args)
    eval = apply(Nx.Defn.jit(fun, @eval), args)
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  # A PRNG key threaded through a decode-style loop: each step splits the key,
  # draws a uniform with one half, keeps the other, and accumulates. Nests
  # threefry's own `while` (and its dynamic rotation-table slice) inside the
  # outer loop body.
  defn sample_loop(key) do
    {_i, _key, acc} =
      while {i = 0, key = key, acc = Nx.tensor(0.0, type: :f32)}, i < 5 do
        keys = Nx.Random.split(key)
        {u, _} = Nx.Random.uniform(keys[1])
        {i + 1, keys[0], acc + u}
      end

    acc
  end

  describe "Nx.Random primitives: native == evaluator (fixed seed)" do
    test "uniform", do: equiv(fn k -> Nx.Random.uniform(k, shape: {8}) |> elem(0) end, [key()])
    test "normal", do: equiv(fn k -> Nx.Random.normal(k, shape: {8}) |> elem(0) end, [key()])

    test "randint",
      do: equiv(fn k -> Nx.Random.randint(k, 0, 100, shape: {8}) |> elem(0) end, [key()])

    test "gumbel", do: equiv(fn k -> Nx.Random.gumbel(k, shape: {8}) |> elem(0) end, [key()])
    test "split", do: equiv(fn k -> Nx.Random.split(k, parts: 4) end, [key()])
  end

  describe "dynamic-start slice (the threefry enabler)" do
    test "slice with a runtime start index matches the evaluator" do
      t = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0], backend: Emily.Backend)
      start = Nx.tensor(1, type: :s32, backend: Emily.Backend)
      out = equiv(fn t, s -> Nx.slice(t, [s], [3]) end, [t, start])
      assert Nx.to_flat_list(out) == [20.0, 30.0, 40.0]
    end

    test "bitcast (random bits -> float) matches the evaluator" do
      bits = Nx.tensor([0, 1_065_353_216, 1_073_741_824], type: :u32, backend: Emily.Backend)
      equiv(fn b -> Nx.bitcast(b, :f32) end, [bits])
    end

    test "out-of-bounds runtime start clamps to [0, dim - length] (Nx semantics)" do
      # Oracle is Nx (BinaryBackend), which clamps the start to dim-length;
      # MLX's dynamic slice would read out of bounds without the clamp.
      data = [10.0, 20.0, 30.0, 40.0, 50.0]
      fun = fn t, s -> Nx.slice(t, [s], [3]) end

      native =
        Nx.Defn.jit(fun, @native).(
          Nx.tensor(data, backend: Emily.Backend),
          Nx.tensor(4, type: :s32, backend: Emily.Backend)
        )

      canonical = Nx.slice(Nx.tensor(data), [Nx.tensor(4)], [3])
      assert Nx.to_flat_list(native) == Nx.to_flat_list(canonical)
      assert Nx.to_flat_list(native) == [30.0, 40.0, 50.0]
    end
  end

  describe "key-splitting sampling while" do
    test "native matches the evaluator bit-for-bit" do
      equiv(&sample_loop/1, [key()])
    end

    test "draws actually depend on the seed (real RNG, not a constant)" do
      a = Nx.Defn.jit(&sample_loop/1, @native).(key(1))
      b = Nx.Defn.jit(&sample_loop/1, @native).(key(2))
      refute Nx.to_binary(a) == Nx.to_binary(b)
    end
  end
end
