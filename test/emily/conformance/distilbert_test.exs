defmodule Emily.Conformance.DistilbertTest do
  @moduledoc """
  End-to-end conformance tests for DistilBERT on `Emily.Backend`.

  These tests port Bumblebee's own DistilBERT test suite
  (`test/bumblebee/text/distilbert_test.exs`) verbatim — same six
  architectures, same tiny-random HuggingFace checkpoints, same inputs,
  same expected output slices. The reference values were produced by the
  HuggingFace Transformers (PyTorch) reference implementation, so any
  divergence here is unambiguously an Emily bug: a bad axis, a wrong
  softmax dim, a transposed matmul.

  A single test module covers the Bumblebee forward path through
  embeddings, 6 transformer blocks (batched self-attention, layer norm,
  GELU FFN), and each of the six task heads. If this suite is green,
  every Nx op on DistilBERT's critical path is correct on Emily.Backend.

  Tagged `:conformance` and excluded from the default suite because the
  tiny-random models are fetched from HuggingFace on first run
  (`~/.cache/bumblebee`). Invoke explicitly:

      mix test --only conformance
  """

  use ExUnit.Case, async: false

  @moduletag :conformance
  @moduletag capture_log: true
  @moduletag timeout: 120_000

  setup_all do
    prev = Nx.default_backend()
    Nx.global_default_backend(Emily.Backend)
    on_exit(fn -> Nx.global_default_backend(prev) end)
    :ok
  end

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DistilBertModel"})

    assert %Bumblebee.Text.Distilbert{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.9427, 0.7933, 0.1031], [1.0913, 1.0214, -1.5890], [-2.1149, -0.3367, -0.6268]]
      ])
    )
  end

  test ":for_masked_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DistilBertForMaskedLM"})

    assert %Bumblebee.Text.Distilbert{architecture: :for_masked_language_modeling} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1124}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[-0.1839, -0.0195, 0.1220], [-0.2048, 0.0667, 0.0878], [-0.2045, -0.0483, -0.1567]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"}
             )

    assert %Bumblebee.Text.Distilbert{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(outputs.logits, Nx.tensor([[-0.0047, -0.0103]]))
  end

  test ":for_token_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DistilBertForTokenClassification"}
             )

    assert %Bumblebee.Text.Distilbert{architecture: :for_token_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 2}

    assert_all_close(
      outputs.logits[[.., 1..3//1, ..]],
      Nx.tensor([[[-0.0504, -0.0751], [0.1354, 0.2180], [-0.0386, 0.1059]]])
    )
  end

  test ":for_question_answering" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DistilBertForQuestionAnswering"}
             )

    assert %Bumblebee.Text.Distilbert{architecture: :for_question_answering} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.start_logits) == {1, 10}
    assert Nx.shape(outputs.end_logits) == {1, 10}

    assert_all_close(
      outputs.start_logits[[.., 1..3]],
      Nx.tensor([[0.1790, -0.0074, 0.0412]])
    )

    assert_all_close(
      outputs.end_logits[[.., 1..3]],
      Nx.tensor([[-0.1520, -0.0973, 0.0166]])
    )
  end

  test ":for_multiple_choice" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "hf-internal-testing/tiny-random-DistilBertForMultipleChoice"}
             )

    assert %Bumblebee.Text.Distilbert{architecture: :for_multiple_choice} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]]),
      "attention_mask" => Nx.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1}

    assert_all_close(outputs.logits, Nx.tensor([[-0.0027]]))
  end

  describe "Nx.Serving.batched_run" do
    # Exercises Bumblebee's question-answering serving end-to-end:
    # tokenizer, forward pass, postprocess, and Nx.Serving's batching
    # pipeline. The tiny-random model produces meaningless answers, so
    # we assert structure rather than content — the point is that the
    # batched path runs cleanly against Emily.Backend.
    test "batched_run drives DistilBERT-QA through Nx.Serving" do
      {:ok, model_info} =
        Bumblebee.load_model(
          {:hf, "hf-internal-testing/tiny-random-DistilBertForQuestionAnswering"}
        )

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "distilbert-base-uncased"})

      serving = Bumblebee.Text.question_answering(model_info, tokenizer)

      start_supervised!({Nx.Serving, serving: serving, name: __MODULE__.Serving})

      inputs = [
        %{question: "What is my name?", context: "My name is Sarah."},
        %{question: "Where do I live?", context: "I live in London."}
      ]

      results = Nx.Serving.batched_run(__MODULE__.Serving, inputs)

      assert length(results) == 2

      for result <- results do
        assert %{results: [%{text: text, score: score, start: s, end: e}]} = result
        assert is_binary(text)
        assert is_float(score)
        assert is_integer(s)
        assert is_integer(e)
      end
    end
  end

  # ----------------- helpers -----------------

  # Mirrors Bumblebee.TestHelpers.assert_all_close: check that all
  # elements agree within (atol + rtol * |right|) after materialising
  # to BinaryBackend (so a mismatch produces a readable inspect diff).
  defp assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equal_tensor =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if Nx.to_number(equal_tensor) != 1 do
      ExUnit.Assertions.flunk("""
      expected

      #{inspect(Nx.backend_copy(left, Nx.BinaryBackend))}

      to be within tolerance of

      #{inspect(Nx.backend_copy(right, Nx.BinaryBackend))}
      """)
    end
  end
end
