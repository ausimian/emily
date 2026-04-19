# Qwen3 conformance suite as a standalone Mix.install script.
#
# Bumblebee's Qwen3 support lives on `main` and hasn't been released
# to Hex yet. Running these tests as part of `mix test` would pin
# Emily's own mix.exs to a github ref, churning the lockfile between
# :test/:docs (github Bumblebee) and :dev/:prod (Hex Bumblebee). This
# script bypasses that by declaring its own dep tree via Mix.install
# — Emily's mix.exs can stay on the Hex Bumblebee and still get
# Qwen3 coverage.
#
# Usage:
#
#     elixir scripts/qwen3_conformance.exs              # tiny-random suite (default)
#     QWEN3_RUN=full elixir scripts/qwen3_conformance.exs    # + full checkpoint
#
# `tiny-random` fetches ~10 MB of HuggingFace checkpoints. `full`
# adds the ~1.2 GB `Qwen/Qwen3-0.6B` fetch.
#
# The quantized variant (formerly `:qwen3_quant_full`) depends on
# `Emily.Quantization.Transform`, which currently lives in
# `test/support/` and so isn't reachable from a Mix.install consumer.
# Once that module graduates to `lib/`, re-introduce a third test
# module here under a `QWEN3_RUN=all` gate.
#
# Switch back to `mix test` integration by bumping Bumblebee on Hex
# (>= whichever release contains `Bumblebee.Text.Qwen3`) and pinning
# that in `mix.exs`; the three ExUnit modules below can then graduate
# back into `test/emily/conformance/`.

Mix.install([
  {:emily, path: Path.expand("..", __DIR__)},
  # `override: true` because Emily's mix.exs declares
  # `{:bumblebee, "~> 0.6", optional: true}` — Mix would otherwise
  # refuse the github ref here as a child-dep conflict.
  {:bumblebee,
   github: "elixir-nx/bumblebee",
   ref: "273805e95507dc7866b958d90e0012a3abad1761",
   override: true},
  {:axon, "~> 0.7"},
  {:tokenizers, "~> 0.5"},
  {:nx, "~> 0.10"}
])

defmodule Qwen3Conf do
  @moduledoc false

  # Inlined from `Emily.ConformanceHelper` — tolerance-aware tensor
  # comparison used in the tiny-random architecture tests.
  def assert_all_close(left, right, opts \\ []) do
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

run_mode = System.get_env("QWEN3_RUN", "tiny")

exclude =
  case run_mode do
    "tiny" -> [:qwen3_full]
    "full" -> []
    other -> raise "QWEN3_RUN must be one of tiny|full, got: #{inspect(other)}"
  end

ExUnit.start(max_cases: System.schedulers_online(), exclude: exclude)

# ---------------------------------------------------------------------
# Tiny-random Qwen3 architectures + greedy decode
# ---------------------------------------------------------------------

defmodule Emily.Conformance.Qwen3Test do
  @moduledoc """
  Mirrors `Bumblebee.Text.Qwen3Test` — same three architectures, same
  tiny-random checkpoints, same input token IDs, same expected output
  slices. A failure here unambiguously indicates an Emily bug on
  Qwen3's critical path: QK-norm, rotary embeddings, GQA, SwiGLU
  FFN, RMSNorm, tied embeddings.

  An additional test drives the `Bumblebee.Text.Generation` pipeline
  greedy-decode; the oracle is `Nx.BinaryBackend` on the same inputs
  (we have no Linux+CUDA EXLA machine in CI), so the assertion is
  really "Emily.Backend and BinaryBackend produce bit-identical
  token ids for the same model under greedy decoding".
  """

  use ExUnit.Case, async: true

  import Qwen3Conf

  alias Bumblebee.Text.Generation, as: BBGeneration

  @moduletag capture_log: true
  @moduletag timeout: 300_000

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

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
    @golden_max_new_tokens 16
    @golden_input_ids [10, 20, 30, 40, 50, 60, 70, 80]
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

# ---------------------------------------------------------------------
# Full-checkpoint greedy decode (~1.2 GB on first fetch)
# ---------------------------------------------------------------------

defmodule Emily.Conformance.Qwen3FullTest do
  @moduledoc false

  use ExUnit.Case, async: true

  @moduletag :qwen3_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  @prompt "The quick brown fox jumps over the lazy dog."
  @reference_text " The quick brown fox is a character in the story. The quick brown fox is a character in the story. The quick brown fox is a character in the story"

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  test "Qwen/Qwen3-0.6B greedy decodes the pinned continuation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-0.6B"})

    config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 32,
        strategy: %{type: :greedy_search}
      )

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, config,
        defn_options: [compiler: Nx.Defn.Evaluator]
      )

    %{results: [%{text: text, token_summary: summary}]} =
      Nx.Serving.run(serving, @prompt)

    assert summary.output == 32
    assert text == @reference_text
  end
end

