defmodule Emily.CompilerAxonTest do
  @moduledoc """
  M5 exit-criterion test: an Axon MLP forward pass under
  `Emily.Compiler` produces results matching `Nx.Defn.Evaluator` on the
  same backend within float tolerance.

  Both compilers execute through `Emily.Backend`; the comparison is
  structural — does our compiler walk the expression and reach the same
  ops with the same operands as the reference walker.
  """

  use ExUnit.Case, async: false

  import Emily.BackendGenerators, only: [assert_close: 2]

  setup do
    prev = Nx.default_backend()
    Nx.default_backend(Emily.Backend)
    on_exit(fn -> Nx.default_backend(prev) end)
    :ok
  end

  test "3-layer MLP forward pass matches Nx.Defn.Evaluator" do
    model =
      Axon.input("input", shape: {nil, 16})
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(16, activation: :relu)
      |> Axon.dense(10)

    {init_fn, predict_fn} = Axon.build(model)

    template = Nx.template({1, 16}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())

    # Deterministic input — we don't care about the values, only that
    # both compilers see the same numbers.
    input =
      Nx.iota({4, 16}, type: :f32)
      |> Nx.divide(64.0)
      |> Nx.subtract(0.5)

    eval =
      Nx.Defn.jit_apply(predict_fn, [params, input], compiler: Nx.Defn.Evaluator)

    emily =
      Nx.Defn.jit_apply(predict_fn, [params, input], compiler: Emily.Compiler)

    assert Nx.shape(emily) == {4, 10}
    assert_close(emily, eval)
  end

  test "Nx.Defn.compile reuse — closure runs many inputs through one walk" do
    model =
      Axon.input("input", shape: {nil, 8})
      |> Axon.dense(8, activation: :tanh)
      |> Axon.dense(4)

    {init_fn, predict_fn} = Axon.build(model)
    template = Nx.template({1, 8}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())

    compiled =
      Nx.Defn.compile(
        predict_fn,
        [params, Nx.template({2, 8}, :f32)],
        compiler: Emily.Compiler
      )

    inputs = [
      Nx.iota({2, 8}, type: :f32),
      Nx.broadcast(0.25, {2, 8}),
      Nx.tensor(for _ <- 1..2, do: for(_ <- 1..8, do: 0.5))
    ]

    for input <- inputs do
      out = compiled.(params, input)
      assert Nx.shape(out) == {2, 4}

      ref =
        Nx.Defn.jit_apply(predict_fn, [params, input], compiler: Nx.Defn.Evaluator)

      assert_close(out, ref)
    end
  end
end
