defmodule Emily.Training.MnistNativeFullTest do
  @moduledoc """
  MNIST MLP convergence under the **native single-NIF compiler**
  (issue #174, `:training_full`).

  The native-lane analogue of `mnist_full_test.exs`: the same dense
  MLP trains on real MNIST, but the whole training step compiles
  through `compiler: Emily.Compiler, native: true, native_fallback:
  :raise` instead of the op-by-op evaluator. Closes the matmul-
  dominated (dense + Adam) half of the training-coverage gap; the
  conv/pooling half is `mnist_cnn_native_full_test.exs`.

  `native_fallback: :raise` makes the run self-proving: once
  `native: true` reaches `Emily.Compiler` the outcome is binary —
  the step lowers to one program and replays in a single NIF, or an
  un-lowerable op raises (it can never silently degrade to the
  evaluator under `:raise`). So reaching the accuracy assertion proves
  the dense forward, cross-entropy loss, backward, and Adam update all
  lowered fully native. See `mnist_cnn_native_full_test.exs` for the
  full rationale.

  Opt-in — `mix test --only training_full`.
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

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]

  test "Axon MLP reaches >#{trunc(@target_accuracy * 100)}% accuracy via the native single-NIF compiler" do
    {train_batches, test_images, test_labels} = MnistHelper.load_mnist(@batch_size)

    model =
      Axon.input("input", shape: {nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dense(10, activation: :softmax)

    trained_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.run(train_batches, %{}, [epochs: @epochs] ++ @native)

    accuracy = MnistHelper.evaluate(model, trained_state, test_images, test_labels, @native)

    assert accuracy >= @target_accuracy,
           "native MNIST MLP accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end
end
