defmodule Emily.MnistHelper do
  @moduledoc false

  def load_mnist(batch_size) do
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

  def evaluate(model, state, test_images, test_labels) do
    logits =
      Axon.predict(model, state, test_images, compiler: Emily.Compiler)

    predicted = Nx.argmax(logits, axis: -1)
    actual = Nx.argmax(test_labels, axis: -1)

    Nx.mean(Nx.equal(predicted, actual))
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_number()
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
end
