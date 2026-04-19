defmodule Emily.CompilerTest do
  @moduledoc """
  Equivalence tests for `Emily.Compiler`.

  Each test runs a computation two ways:

    * the same anonymous function applied directly under
      `Emily.Backend` (no `defn`), serving as the oracle, and
    * the same function jitted under `Emily.Compiler`.

  The two paths use the same MLX kernels for the underlying ops, so the
  oracle is structural ("did the compiler walk the expression and reach
  the right backend?") rather than numerical. The Backend's own
  property suite already verifies the kernels themselves against
  `Nx.BinaryBackend`.

  `defn`-only constructs (`while`, `cond`, `Nx.Defn.Kernel.transform/2`)
  get an explicit `defn` definition and run under both `Emily.Compiler`
  and `Nx.Defn.Evaluator` to verify equivalence.
  """

  use ExUnit.Case, async: true

  import Nx.Defn
  import Emily.BackendGenerators, only: [assert_close: 2]

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  defp jit_emily(fun, args), do: Nx.Defn.jit_apply(fun, args, compiler: Emily.Compiler)

  # ------------------------------------------------------------------
  # Behaviour callbacks
  # ------------------------------------------------------------------

  describe "callback contracts" do
    test "__to_backend__ returns Emily.Backend with default device :gpu" do
      assert {Emily.Backend, [device: :gpu]} == Emily.Compiler.__to_backend__([])
    end

    test "__to_backend__ honours :device opt" do
      assert {Emily.Backend, [device: :cpu]} == Emily.Compiler.__to_backend__(device: :cpu)
    end

    test "__partitions_options__ pins to a single partition" do
      assert [[]] == Emily.Compiler.__partitions_options__([])
      assert [[max_concurrency: 1]] == Emily.Compiler.__partitions_options__(max_concurrency: 1)
    end

    test "__partitions_options__ rejects max_concurrency > 1" do
      assert_raise ArgumentError, ~r/does not support :max_concurrency > 1/, fn ->
        Emily.Compiler.__partitions_options__(max_concurrency: 4)
      end
    end

    test "rejects unknown options" do
      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Emily.Compiler.__to_backend__(bogus: true)
      end
    end

    # Nx.Serving prepends :batch_keys to defn_options for arity-1
    # serving builders (e.g. Bumblebee's speech_to_text_whisper/5);
    # Bumblebee's Shared.compile_or_jit propagates :cache.
    # Evaluator ignores both, but rejecting them at validate time
    # breaks those flows. Regression for the Whisper notebook smoke
    # test.
    test "accepts :batch_keys and :cache without raising" do
      assert [[batch_keys: [:default]]] ==
               Emily.Compiler.__partitions_options__(batch_keys: [:default])

      assert {Emily.Backend, [device: :gpu]} ==
               Emily.Compiler.__to_backend__(batch_keys: [:default], cache: "bumblebee")
    end

    test "jit passes through :batch_keys without raising" do
      fun = fn x -> Nx.add(x, 1.0) end

      result =
        Nx.Defn.jit_apply(fun, [Nx.tensor([1.0, 2.0])],
          compiler: Emily.Compiler,
          batch_keys: [:default]
        )

      assert_close(result, Nx.tensor([2.0, 3.0]))
    end
  end

  # ------------------------------------------------------------------
  # Op equivalence (jit vs raw Backend)
  # ------------------------------------------------------------------

  describe "elementwise ops" do
    test "add" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      direct = Nx.add(a, b)
      jitted = jit_emily(fn x, y -> Nx.add(x, y) end, [a, b])
      assert_close(jitted, direct)
    end

    test "chain of binary + unary" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[0.5, 1.5], [2.5, 3.5]])
      fun = fn x, y -> x |> Nx.multiply(y) |> Nx.exp() |> Nx.add(1.0) end
      assert_close(jit_emily(fun, [a, b]), fun.(a, b))
    end

    test "broadcasting" do
      a = Nx.tensor([[1.0, 2.0, 3.0]])
      b = Nx.tensor([[1.0], [2.0], [3.0]])
      fun = fn x, y -> Nx.add(x, y) end
      assert_close(jit_emily(fun, [a, b]), fun.(a, b))
    end
  end

  describe "reductions" do
    test "sum no axes" do
      t = Nx.iota({3, 4}, type: :f32)
      fun = fn x -> Nx.sum(x) end
      assert_close(jit_emily(fun, [t]), fun.(t))
    end

    test "sum with axis + keepdims" do
      t = Nx.iota({2, 3, 4}, type: :f32)
      fun = fn x -> Nx.sum(x, axes: [1], keep_axes: true) end
      assert_close(jit_emily(fun, [t]), fun.(t))
    end

    test "argmax" do
      t = Nx.tensor([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]])
      fun = fn x -> Nx.argmax(x, axis: 1) end
      jitted = jit_emily(fun, [t])
      direct = fun.(t)
      assert Nx.to_flat_list(jitted) == Nx.to_flat_list(direct)
    end
  end

  describe "shape ops" do
    test "reshape + transpose" do
      t = Nx.iota({2, 3, 4}, type: :f32)
      fun = fn x -> x |> Nx.reshape({6, 4}) |> Nx.transpose() end
      assert_close(jit_emily(fun, [t]), fun.(t))
    end

    test "concatenate" do
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[3.0, 4.0]])
      fun = fn x, y -> Nx.concatenate([x, y], axis: 0) end
      assert_close(jit_emily(fun, [a, b]), fun.(a, b))
    end
  end

  describe "linalg" do
    @describetag :linalg
    test "dot product (matmul)" do
      a = Nx.iota({3, 4}, type: :f32)
      b = Nx.iota({4, 2}, type: :f32)
      fun = fn x, y -> Nx.dot(x, y) end
      assert_close(jit_emily(fun, [a, b]), fun.(a, b))
    end
  end

  # ------------------------------------------------------------------
  # Container outputs
  # ------------------------------------------------------------------

  describe "container outputs" do
    test "tuple result" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      fun = fn x -> {Nx.sum(x), Nx.mean(x)} end
      {sum, mean} = jit_emily(fun, [t])
      assert_close(sum, Nx.sum(t))
      assert_close(mean, Nx.mean(t))
    end
  end

  # ------------------------------------------------------------------
  # defn-only constructs: while + cond
  # ------------------------------------------------------------------

  defn loop_add_one(x, n) do
    {result, _, _} =
      while {acc = x, i = 0, n}, Nx.less(i, n) do
        {Nx.add(acc, 1.0), Nx.add(i, 1), n}
      end

    result
  end

  defn cond_branch(x) do
    cond do
      Nx.greater(x, 0.0) -> Nx.multiply(x, 2.0)
      Nx.equal(x, 0.0) -> Nx.tensor(0.0)
      true -> Nx.negate(x)
    end
  end

  describe "control flow under defn" do
    test "while loop matches Evaluator" do
      x = Nx.tensor(10.0)
      n = Nx.tensor(5)

      emily =
        Nx.Defn.jit_apply(&loop_add_one/2, [x, n], compiler: Emily.Compiler)

      eval =
        Nx.Defn.jit_apply(&loop_add_one/2, [x, n], compiler: Nx.Defn.Evaluator)

      assert_close(emily, eval)
      assert Nx.to_number(emily) == 15.0
    end

    test "cond chooses the correct branch" do
      for x_val <- [1.5, 0.0, -2.0] do
        x = Nx.tensor(x_val)
        emily = Nx.Defn.jit_apply(&cond_branch/1, [x], compiler: Emily.Compiler)
        eval = Nx.Defn.jit_apply(&cond_branch/1, [x], compiler: Nx.Defn.Evaluator)
        assert_close(emily, eval)
      end
    end
  end

  # ------------------------------------------------------------------
  # Closure reuse — Nx.Defn.compile path
  # ------------------------------------------------------------------

  describe "Nx.Defn.compile/3 reuse" do
    test "compiled closure executes repeatedly with the same shape" do
      template = Nx.template({4}, :f32)

      compiled =
        Nx.Defn.compile(fn x -> Nx.multiply(x, 2.0) end, [template], compiler: Emily.Compiler)

      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      b = Nx.tensor([5.0, 6.0, 7.0, 8.0])

      assert_close(compiled.(a), Nx.tensor([2.0, 4.0, 6.0, 8.0]))
      assert_close(compiled.(b), Nx.tensor([10.0, 12.0, 14.0, 16.0]))
    end
  end
end
