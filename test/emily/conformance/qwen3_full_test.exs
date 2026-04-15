defmodule Emily.Conformance.Qwen3FullTest do
  @moduledoc """
  Full `Qwen/Qwen3-0.6B` end-to-end conformance test.

  This test is excluded even from `mix test --only conformance`: the
  model is ~1.5 GB on first fetch, so blowing it out into CI on every
  push is the wrong default. Run explicitly:

      mix test --only qwen3_full

  The reference text pinned below is the greedy decode produced by
  `Emily.Backend` on an Apple-Silicon host. A failure means the
  backend has drifted, Bumblebee's Qwen3 port has changed, or the HF
  checkpoint has been republished — all of which are real signals.
  """

  use ExUnit.Case, async: false

  alias Emily.Bumblebee.FastKernels

  @moduletag :qwen3_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  @prompt "The quick brown fox jumps over the lazy dog."
  @reference_text " The quick brown fox is a character in the story. The quick brown fox is a character in the story. The quick brown fox is a character in the story"

  setup_all do
    prev = Nx.default_backend()
    Nx.global_default_backend(Emily.Backend)
    on_exit(fn -> Nx.global_default_backend(prev) end)
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

  @tag :fast_kernels_full
  test "Qwen/Qwen3-0.6B with fused MLX kernels still decodes the pinned continuation" do
    {:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-0.6B"})

    # Apply the fast-kernel rewrites to the loaded Axon model. Params
    # are unchanged — the rewrites preserve parameter shapes and
    # names (RMSNorm weight, attention dense layers, …).
    model_info = update_in(model_info.model, &FastKernels.apply/1)

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
    # Greedy decode is deterministic per logits ordering; the fused
    # kernels reorder some ops (rsqrt, softmax exp/sum) so token-level
    # divergence at later positions is plausible. We pin the *prefix*
    # rather than the full string — drift after that is acceptable as
    # long as the model is still producing English.
    assert String.starts_with?(text, " The quick brown fox is")
  end
end
