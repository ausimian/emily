defmodule Emily.Conformance.Qwen3Test do
  @moduledoc """
  End-to-end conformance tests for Qwen3 on `Emily.Backend`.

  Mirrors `Bumblebee.Text.Qwen3Test` — same three architectures, same
  tiny-random checkpoints, same input token IDs, same expected output
  slices. The reference values in Bumblebee's own test suite were
  produced by the HuggingFace Transformers reference implementation, so
  a failure here unambiguously indicates an Emily bug on Qwen3's critical
  path: QK-norm, rotary embeddings, GQA, SwiGLU FFN, RMSNorm, tied
  embeddings.

  An additional test drives the `Bumblebee.Text.Generation` pipeline
  greedy-decode with a fixed prompt; the token sequence is checked in
  after validation against `Nx.BinaryBackend` on the same inputs.
  `BinaryBackend` is the conformance-layer oracle here (we don't have a
  Linux+CUDA EXLA machine in CI), so the assertion is really
  "Emily.Backend and BinaryBackend produce bit-identical token ids for
  the same model under greedy decoding".

  Tagged `:conformance` and excluded from the default suite; the
  tiny-random checkpoints are fetched from HuggingFace on first run and
  cached under `~/.cache/bumblebee`. Invoke explicitly:

      mix test --only conformance
  """

  use ExUnit.Case, async: false
  use Emily.ConformanceHelper

  alias Bumblebee.Text.Generation, as: BBGeneration

  @moduletag :conformance
  @moduletag capture_log: true
  @moduletag timeout: 300_000

  test ":base" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3Model"})

    assert %Bumblebee.Text.Qwen3{architecture: :base} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 10, 32}

    assert_all_close(
      outputs.hidden_state[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.0461, -1.5811, 1.5504], [0.1340, -1.3477, 0.8047], [-0.5821, -0.4164, 0.9769]]
      ])
    )
  end

  test ":for_causal_language_modeling" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM"})

    assert %Bumblebee.Text.Qwen3{architecture: :for_causal_language_modeling} = spec
    assert spec.use_qk_norm == true

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 10, 1024}

    assert_all_close(
      outputs.logits[[.., 1..3, 1..3]],
      Nx.tensor([
        [[0.1457, 0.0313, -0.0651], [0.1718, -0.0265, -0.0186], [0.2281, -0.0124, -0.0147]]
      ])
    )
  end

  test ":for_sequence_classification" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model(
               {:hf, "bumblebee-testing/tiny-random-Qwen3ForSequenceClassification"}
             )

    assert %Bumblebee.Text.Qwen3{architecture: :for_sequence_classification} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 2}

    assert_all_close(outputs.logits, Nx.tensor([[-0.1487, -0.0071]]))
  end

  describe "greedy generation" do
    # Exercises the full `Bumblebee.Text.Generation` path on Emily.Backend:
    # Axon forward pass + logit processing + argmax + `put_slice` KV
    # cache updates + `while` loop in defn. The tiny-random causal LM
    # emits gibberish tokens, but determinism is the whole point here
    # — the oracle is `Nx.BinaryBackend` running the identical generate
    # function against the identical token ids, and we assert bit-exact
    # token equality.
    #
    # BinaryBackend is our reference because we do not have a Linux+CUDA
    # EXLA machine in CI (see PLAN.md M3/M4 testing philosophy).
    #
    # We bypass Bumblebee's serving so that the tokenizer isn't involved:
    # the tiny-random checkpoint carries a 1024-token embedding while the
    # Qwen3 tokenizer produces ids up to 151_936, so any real prompt
    # would out-of-bounds the embedding gather. The synthetic ids below
    # are all < 1024 and stand in for a pre-tokenized prompt.

    @golden_max_new_tokens 16
    @golden_input_ids [10, 20, 30, 40, 50, 60, 70, 80]

    # Reference produced by `Nx.BinaryBackend` on the same inputs; also
    # cross-checked against BinaryBackend at test time below. If this
    # list changes you have either drifted one backend, drifted
    # Bumblebee's Qwen3 port, or upgraded to a new tiny-random
    # checkpoint — none of which should happen silently.
    @golden_tokens [6, 277, 436, 806, 833, 436, 785, 135, 550, 309, 511, 89, 865, 72, 1021, 865]

    test "tiny-random Qwen3ForCausalLM: greedy decode matches checked-in golden tokens" do
      emily_tokens = run_greedy(Emily.Backend)

      oracle_tokens =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          run_greedy(Nx.BinaryBackend)
        end)

      assert length(emily_tokens) == @golden_max_new_tokens
      assert emily_tokens == oracle_tokens, "Emily.Backend diverged from BinaryBackend"
      assert emily_tokens == @golden_tokens, "both backends drifted from checked-in reference"
    end

    defp run_greedy(backend) do
      {:ok, %{model: model, params: params, spec: spec}} =
        Bumblebee.load_model(
          {:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM"},
          backend: backend
        )

      config =
        Bumblebee.configure(
          %Bumblebee.Text.GenerationConfig{
            pad_token_id: 0,
            eos_token_id: 0
          },
          max_new_tokens: @golden_max_new_tokens,
          strategy: %{type: :greedy_search}
        )

      generate_fun = BBGeneration.build_generate(model, spec, config)

      input_ids = Nx.tensor([@golden_input_ids], type: :s64, backend: backend)
      attention_mask = Nx.broadcast(Nx.tensor(1, backend: backend), Nx.shape(input_ids))
      seed = Nx.tensor([0], type: :s64, backend: backend)

      inputs = %{
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "seed" => seed
      }

      %{token_ids: token_ids, length: length} =
        Nx.Defn.jit_apply(generate_fun, [params, inputs], compiler: Nx.Defn.Evaluator)

      tokens = token_ids |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
      [len_val] = length |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
      Enum.take(tokens, len_val)
    end
  end
end
