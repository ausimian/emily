defmodule Emily.Training.MnistFullTest do
  @moduledoc """
  MNIST convergence canary (M9 Phase F, `:training_full`).

  Opt-in — `mix test --only training_full`. Downloads MNIST via
  `scidata` (~11 MB, cached in `~/Library/Caches/scidata`), trains an
  Axon MLP on `Emily.Compiler`, and asserts >96% test accuracy.

  Purpose: catch systemic numerical drift that the curve-matching
  tests in `mlp_curve_test.exs` can't see — those compare Emily vs
  `Nx.BinaryBackend` on the same synthetic task, so a shared bug in
  `Nx.Defn.grad`'s symbolic rules would be invisible. Training on
  real data to a real accuracy target is a cross-check: if Emily
  converges on MNIST the way PyTorch/JAX do, the grad chain is
  numerically sound end-to-end.

  Not a regression-gate on performance — just "did it converge".
  Runs once per invocation, no property sweep.
  """

  use ExUnit.Case, async: true

  alias Emily.MnistHelper

  @moduletag :training_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  @batch_size 128
  @epochs 5
  @target_accuracy 0.96

  test "Axon MLP reaches >#{trunc(@target_accuracy * 100)}% test accuracy under Emily.Compiler" do
    {train_batches, test_images, test_labels} = MnistHelper.load_mnist(@batch_size)

    model =
      Axon.input("input", shape: {nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dense(10, activation: :softmax)

    trained_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.run(train_batches, %{},
        epochs: @epochs,
        compiler: Emily.Compiler
      )

    accuracy = MnistHelper.evaluate(model, trained_state, test_images, test_labels)

    assert accuracy >= @target_accuracy,
           "MNIST accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end
end
