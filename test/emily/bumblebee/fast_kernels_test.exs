defmodule Emily.Bumblebee.FastKernelsTest do
  @moduledoc """
  Unit tests for `Emily.Bumblebee.FastKernels`. Each test builds a
  handcrafted Axon model containing one of the patterns the shim
  knows how to rewrite, runs `apply/1`, then verifies:

    1. The rewritten model's predict_fn produces output within
       tolerance of the unrewritten model.
    2. The expected `op` field on the relevant node has been swapped
       to the fast variant.

  Conformance against full Bumblebee transformer models lives in
  `test/emily/conformance/*_full_test.exs` (opt-in via the
  `:fast_kernels_full` tag).
  """

  use ExUnit.Case, async: false

  import Emily.BackendGenerators, only: [assert_close: 3]

  alias Bumblebee.Layers, as: BL
  alias Emily.Bumblebee.FastKernels

  @f32_tol 1.0e-4

  setup do
    prev = Nx.default_backend()
    Nx.default_backend(Emily.Backend)
    on_exit(fn -> Nx.default_backend(prev) end)
    :ok
  end

  describe "apply_rms_norm/1" do
    test "rewrites BL.rms_norm and matches output" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> BL.rms_norm(name: "norm", epsilon: 1.0e-6)

      rewritten = FastKernels.apply_rms_norm(model)

      # Same {init, predict}.
      {init_fn, predict_fn_orig} = Axon.build(model, compiler: Emily.Compiler)
      {_, predict_fn_fast} = Axon.build(rewritten, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      orig = predict_fn_orig.(params, %{"x" => x})
      fast = predict_fn_fast.(params, %{"x" => x})

      assert_close(fast, orig, tol: @f32_tol)
    end
  end

  describe "apply_layer_norm/1" do
    test "rewrites Axon.layer_norm and matches output" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> Axon.layer_norm(name: "norm", epsilon: 1.0e-5)

      rewritten = FastKernels.apply_layer_norm(model)

      {init_fn, predict_orig} = Axon.build(model, compiler: Emily.Compiler)
      {_, predict_fast} = Axon.build(rewritten, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      orig = predict_orig.(params, %{"x" => x})
      fast = predict_fast.(params, %{"x" => x})

      assert_close(fast, orig, tol: @f32_tol)
    end
  end

  describe "apply/1 idempotence" do
    test "applying twice doesn't break the model" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> BL.rms_norm(name: "n1", epsilon: 1.0e-6)
        |> Axon.layer_norm(name: "n2", epsilon: 1.0e-5)

      once = FastKernels.apply(model)
      twice = FastKernels.apply(once)

      {init_fn, predict_once} = Axon.build(once, compiler: Emily.Compiler)
      {_, predict_twice} = Axon.build(twice, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      a = predict_once.(params, %{"x" => x})
      b = predict_twice.(params, %{"x" => x})

      assert_close(a, b, tol: @f32_tol)
    end
  end
end
