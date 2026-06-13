defmodule Emily.Training.MnistCnnNativeFullTest do
  @moduledoc """
  MNIST CNN convergence under the **native single-NIF compiler**
  (issue #174, `:training_full`).

  The native-lane analogue of `mnist_cnn_full_test.exs`: the same
  LeNet-style Axon CNN trains on real MNIST, but the whole training
  step compiles through `compiler: Emily.Compiler, native: true,
  native_fallback: :raise` instead of the op-by-op evaluator.

  Why `native_fallback: :raise` makes this self-proving. Once
  `native: true` reaches `Emily.Compiler`, the run is binary: the Expr
  lowers to one program and replays in a single NIF, or an op it can't
  lower raises (`Emily.Compiler.build_native/4` reraises the lowering
  `ArgumentError` under `:raise` — it can never silently degrade to the
  evaluator, that path only exists under `:eval`). So a training run
  that *completes* proves the entire step — forward (conv, ReLU,
  maxpool), categorical-cross-entropy loss, the backward
  (`window_scatter_max` for the maxpool grad, `reverse` for the conv
  kernel flip), and the Adam update — lowered fully native with zero
  fallback. `Axon.Loop.run` forwards the per-call jit options (it pops
  only `:jit_compile?`/`:force_garbage_collection?`), so the options
  genuinely reach the compiler.

  Native replay is bit-identical to the evaluator (same MLX kernels in
  the same order), so the accuracy bar matches the eval canary exactly
  (>97%). The point isn't a different number — it's that training
  reaches it through the single-NIF path.

  Opt-in — `mix test --only training_full` (downloads MNIST, multi-
  minute training).
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

  # Strict no-fallback native lane. `:raise` makes a completed run a
  # proof of full native lowering (see the moduledoc).
  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]

  test "Axon CNN reaches >#{trunc(@target_accuracy * 100)}% accuracy via the native single-NIF compiler" do
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

    # The whole training loop (init + step) compiles native: `Axon.Loop.run`
    # forwards `native:`/`native_fallback:` to the defn jit. Under `:raise`,
    # reaching the end proves every op lowered — no silent eval fallback.
    trained_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.run(train_batches, %{}, [epochs: @epochs] ++ @native)

    # Evaluate through the native path too, so the accuracy that gates the
    # test is itself produced by the single-NIF forward.
    accuracy = MnistHelper.evaluate(model, trained_state, test_images, test_labels, @native)

    assert accuracy >= @target_accuracy,
           "native MNIST CNN accuracy #{Float.round(accuracy, 4)} below target #{@target_accuracy}"
  end
end
