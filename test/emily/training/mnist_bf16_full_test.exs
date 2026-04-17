defmodule Emily.Training.MnistBf16FullTest do
  @moduledoc """
  bf16 MNIST convergence canary (M16, `:training_full`).

  Same architecture and training setup as `MnistFullTest`, but with
  Axon's mixed-precision policy: `params: {:f, 32}, compute: {:bf, 16},
  output: {:f, 32}`. Target accuracy is 95.5% — 0.5% below the f32
  baseline (96%), accounting for bf16's reduced precision.

  Opt-in — `mix test --only training_full`.
  """

  use ExUnit.Case, async: false

  @moduletag :training_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  setup_all do
    prev = Nx.default_backend()
    Nx.global_default_backend(Emily.Backend)
    on_exit(fn -> Nx.global_default_backend(prev) end)
    :ok
  end

  @batch_size 128
  @epochs 5
  @target_accuracy 0.955

  test "bf16 Axon MLP reaches >#{trunc(@target_accuracy * 100)}% test accuracy under Emily.Compiler" do
    {train_batches, test_images, test_labels} = load_mnist(@batch_size)

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

    accuracy = evaluate(model, trained_state, test_images, test_labels)

    assert accuracy >= @target_accuracy,
           "bf16 MNIST accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end

  defp load_mnist(batch_size) do
    {train_images_raw, train_labels_raw} = Scidata.MNIST.download()

    train_images =
      train_images_raw
      |> mnist_images_to_tensor()
      |> Nx.to_batched(batch_size)

    train_labels =
      train_labels_raw
      |> mnist_labels_to_tensor()
      |> Nx.to_batched(batch_size)

    train_batches = Stream.zip(train_images, train_labels)

    {test_images_raw, test_labels_raw} = Scidata.MNIST.download_test()

    test_images = mnist_images_to_tensor(test_images_raw)
    test_labels = mnist_labels_to_tensor(test_labels_raw)

    {train_batches, test_images, test_labels}
  end

  defp mnist_images_to_tensor({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
  end

  defp mnist_labels_to_tensor({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.iota({1, 10}))
  end

  defp evaluate(model, state, test_images, test_labels) do
    logits =
      Axon.predict(model, state, test_images, compiler: Emily.Compiler)

    predicted = Nx.argmax(logits, axis: -1)
    actual = Nx.argmax(test_labels, axis: -1)

    Nx.mean(Nx.equal(predicted, actual))
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_number()
  end
end
