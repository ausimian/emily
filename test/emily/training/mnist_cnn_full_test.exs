defmodule Emily.Training.MnistCnnFullTest do
  @moduledoc """
  MNIST CNN convergence canary (M17 exit gate, `:training_full`).

  Opt-in — `mix test --only training_full`. Trains a LeNet-style Axon
  CNN on MNIST under `Emily.Compiler` and asserts >97% test accuracy.

  This is the regression gate for M17's native window ops: Axon's
  `max_pool` layer lowers to `Nx.window_max` on the forward pass and
  to `Nx.window_scatter_max` on the backward. If either path has a
  numerics bug that the synthetic `cnn_curve_test.exs` doesn't catch
  (e.g. a systemic bias that averages out at small scale), this test
  will fail to converge on real data.

  Higher accuracy target than the MLP canary (>97% vs >96%) because a
  CNN with 2x2 pooling and ReLU reliably clears that bar within 5
  epochs on MNIST.
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

  @batch_size 64
  @epochs 5
  @target_accuracy 0.97

  test "Axon CNN reaches >#{trunc(@target_accuracy * 100)}% test accuracy under Emily.Compiler" do
    {train_batches, test_images, test_labels} = MnistHelper.load_mnist(@batch_size, :cnn)

    # Channels-last (Axon default) — MnistHelper produces {N, 28, 28, 1}.
    model =
      Axon.input("input", shape: {nil, 28, 28, 1})
      |> Axon.conv(8, kernel_size: {3, 3}, activation: :relu)
      |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2])
      |> Axon.conv(16, kernel_size: {3, 3}, activation: :relu)
      |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2])
      |> Axon.flatten()
      |> Axon.dense(64, activation: :relu)
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
           "MNIST CNN accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end
end
