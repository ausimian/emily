defmodule Emily.CompilerWhileTest do
  @moduledoc """
  CM10 — `defn while` compiles to the single-NIF replay, bit-identical to
  the Evaluator. The loop body and condition are lowered to nested
  sub-programs (their loop-carried state bound as inputs); the worker thread
  runs the loop, evaluating the condition before each body step
  (`mx::eval` + `item`), so a data-dependent trip count works without any
  BEAM round-trip per iteration. The `while` instruction is multi-output;
  `:elem` projects the final state.

  Run under `native_fallback: :raise` so anything that fails to lower raises
  rather than passing via the evaluator — these are coverage gates.
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]
  # CM14: the opt-in fused-while lane — the host-controlled decode loop with
  # each loop *body* fused under `mx::compile`. The fusion reassociates f32,
  # so this is all-close, not bit-identical (asserted with a tolerance below).
  @fuse [
    compiler: Emily.Compiler,
    native: true,
    native_fallback: :raise,
    fuse: true
  ]

  defp t(data), do: Nx.tensor(data, backend: Emily.Backend)

  # Compute the maximum absolute difference between two flat lists of numbers.
  defp drift(a, b) do
    Enum.zip(a, b)
    |> Enum.reduce(0.0, fn {x, y}, acc -> max(acc, abs(x - y)) end)
  end

  defp equiv(fun, args) do
    native = apply(Nx.Defn.jit(fun, @native), args)
    eval = apply(Nx.Defn.jit(fun, @eval), args)
    assert %Emily.Backend{} = native.data
    assert native.shape == eval.shape and native.type == eval.type
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  # Trip count depends on the runtime values (sum grows past a threshold) —
  # the loop genuinely runs a data-dependent number of times.
  defn count_until(x) do
    {_i, acc} =
      while {i = 0, acc = x}, Nx.less(Nx.sum(acc), 50.0) do
        {i + 1, Nx.multiply(acc, 1.3)}
      end

    acc
  end

  # Condition false at entry -> zero iterations -> returns the initial state.
  defn zero_iter(x) do
    {_i, acc} =
      while {i = 10, acc = x}, Nx.less(i, 5) do
        {i + 1, Nx.multiply(acc, 2.0)}
      end

    acc
  end

  # Single-tensor (non-tuple) loop state: the `while` is used directly, with
  # no `:elem` projection.
  defn double_until(x) do
    while acc = x, Nx.less(Nx.reduce_max(acc), 100.0) do
      Nx.multiply(acc, 2.0)
    end
  end

  # Generation-shaped: a dynamic `Nx.put_slice` at the runtime offset `i`
  # inside the loop body (the same shape as a KV-cache write).
  defn fill_buffer(buf0) do
    {_i, buf} =
      while {i = 0, buf = buf0}, Nx.less(i, 4) do
        val = Nx.reshape(Nx.add(Nx.as_type(i, :f32), 1.0), {1})
        {i + 1, Nx.put_slice(buf, [i], val)}
      end

    buf
  end

  # A loop whose body runs a softmax over the last axis before folding it
  # back into the state. The softmax's elementwise run (max/sub/exp/sum/div)
  # is exactly what `mx::compile` fuses, so this exercises the fused-while
  # body — and the reassociation that makes it all-close rather than exact.
  defn loop_softmax(x) do
    {_i, acc} =
      while {i = 0, acc = x}, Nx.less(i, 8) do
        m = Nx.reduce_max(acc, axes: [-1], keep_axes: true)
        e = Nx.exp(Nx.subtract(acc, m))
        w = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
        {i + 1, Nx.add(acc, w)}
      end

    acc
  end

  describe "fused-while (fuse) == evaluator within f32 tol" do
    test "loop body with a softmax run fuses and stays all-close" do
      x = t([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])

      fused = Nx.Defn.jit(&loop_softmax/1, @fuse).(x)
      eval = Nx.Defn.jit(&loop_softmax/1, @eval).(x)

      assert %Emily.Backend{} = fused.data
      assert fused.shape == eval.shape and fused.type == eval.type

      # mx::compile reassociates the fused elementwise run, so the result
      # matches the evaluator to within a few ULP rather than bit-for-bit.
      drift = drift(Nx.to_flat_list(fused), Nx.to_flat_list(eval))

      assert drift <= 1.0e-5, "fused-while vs evaluator drift #{drift} exceeds 1.0e-5"
    end

    test "data-dependent trip count is honoured under fusion" do
      # The fused body must not change the host-controlled loop's behaviour:
      # the condition is still evaluated each step, so a varying trip count
      # still tracks the input (same property as the plain native lane).
      for data <- [[1.0, 1.0], [5.0, 5.0], [20.0, 20.0]] do
        fused = Nx.Defn.jit(&count_until/1, @fuse).(t(data))
        eval = Nx.Defn.jit(&count_until/1, @eval).(t(data))

        drift = drift(Nx.to_flat_list(fused), Nx.to_flat_list(eval))

        assert drift <= 1.0e-4, "fused-while drift #{drift} exceeds tol for #{inspect(data)}"
      end
    end

    test "zero iterations returns the initial state unchanged (fused lane)" do
      x = t([3.0, 4.0])
      out = Nx.Defn.jit(&zero_iter/1, @fuse).(x)
      # No body ran, so nothing was fused — bit-identical to the input.
      assert Nx.to_binary(out) == Nx.to_binary(x)
    end

    test "dynamic put_slice (KV-write shape) body fuses correctly" do
      # The generation-shaped body: a `Nx.put_slice` at the runtime loop
      # offset — the same dynamic write a KV-cache update lowers to, and the
      # one mx::compile must trace correctly for native generation. The body
      # writes exact integer-valued floats (no reassociation), so the fused
      # result is bit-identical to both the evaluator and the plain native
      # lane here, which pins the dynamic-slice-under-fusion path.
      buf0 = Nx.broadcast(t(0.0), {4})
      fused = Nx.Defn.jit(&fill_buffer/1, @fuse).(buf0)
      eval = Nx.Defn.jit(&fill_buffer/1, @eval).(buf0)

      assert %Emily.Backend{} = fused.data
      assert Nx.to_flat_list(fused) == [1.0, 2.0, 3.0, 4.0]
      assert Nx.to_binary(fused) == Nx.to_binary(eval)
    end
  end

  describe "defn while compiles native == evaluator" do
    test "data-dependent trip count" do
      for data <- [[1.0, 1.0], [5.0, 5.0], [20.0, 20.0]] do
        equiv(&count_until/1, [t(data)])
      end
    end

    test "data-dependent trip count actually varies with the input" do
      small = Nx.Defn.jit(&count_until/1, @native).(t([1.0, 1.0]))
      large = Nx.Defn.jit(&count_until/1, @native).(t([20.0, 20.0]))
      # Different inputs hit the threshold after different iteration counts,
      # so the results are not the same scaling.
      refute Nx.to_binary(small) == Nx.to_binary(large)
    end

    test "zero iterations returns the initial state unchanged" do
      x = t([3.0, 4.0])
      out = equiv(&zero_iter/1, [x])
      assert Nx.to_binary(out) == Nx.to_binary(x)
    end

    test "single-tensor loop state (no :elem projection)" do
      equiv(&double_until/1, [t([1.0, 3.0, 7.0])])
    end

    test "dynamic put_slice (KV-write shape) inside the loop body" do
      out = equiv(&fill_buffer/1, [Nx.broadcast(t(0.0), {4})])
      assert Nx.to_flat_list(out) == [1.0, 2.0, 3.0, 4.0]
    end
  end
end
