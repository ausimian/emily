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
  @target_accuracy 0.96

  test "Axon MLP reaches >#{trunc(@target_accuracy * 100)}% test accuracy under Emily.Compiler" do
    {train_batches, test_images, test_labels} = load_mnist(@batch_size)

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

    accuracy = evaluate(model, trained_state, test_images, test_labels)

    assert accuracy >= @target_accuracy,
           "MNIST accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end

  # ---- Data loading ----

  defp load_mnist(batch_size) do
    # Training set — streamed in batches for Axon.Loop.
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

    # Test set — loaded as whole tensors for one-shot evaluation.
    {test_images_raw, test_labels_raw} = Scidata.MNIST.download_test()

    test_images = mnist_images_to_tensor(test_images_raw)
    test_labels = mnist_labels_to_tensor(test_labels_raw)

    {train_batches, test_images, test_labels}
  end

  defp mnist_images_to_tensor({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
    # {N, 1, 28, 28} → {N, 784} + normalize to [0, 1].
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

  # ---- Evaluation ----

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
