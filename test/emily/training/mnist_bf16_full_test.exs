defmodule Emily.Training.MnistBf16FullTest do
  @moduledoc """
  bf16 MNIST convergence canary (M16, `:training_full`).

  Same architecture and training setup as `MnistFullTest`, but with
  Axon's mixed-precision policy: `params: {:f, 32}, compute: {:bf, 16},
  output: {:f, 32}`. Target accuracy is 95.5% — 0.5% below the f32
  baseline (96%), accounting for bf16's reduced precision.

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
  @target_accuracy 0.955

  test "bf16 Axon MLP reaches >#{trunc(@target_accuracy * 100)}% test accuracy under Emily.Compiler" do
    {train_batches, test_images, test_labels} = MnistHelper.load_mnist(@batch_size)

    policy =
      Axon.MixedPrecision.create_policy(
        params: {:f, 32},
        compute: {:bf, 16},
        output: {:f, 32}
      )

    model =
      Axon.input("input", shape: {nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dense(10, activation: :softmax)
      |> Axon.MixedPrecision.apply_policy(policy)

    trained_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.run(train_batches, %{},
        epochs: @epochs,
        compiler: Emily.Compiler
      )

    accuracy = MnistHelper.evaluate(model, trained_state, test_images, test_labels)

    assert accuracy >= @target_accuracy,
           "bf16 MNIST accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end
end
