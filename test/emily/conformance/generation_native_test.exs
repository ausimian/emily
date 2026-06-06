defmodule Emily.Conformance.GenerationNativeTest do
  @moduledoc """
  CM13 — the goal gate: `Bumblebee.Text.generation`'s own decode loop (a
  `defn while`) compiles **fully native** under `compiler: Emily.Compiler,
  native: true`, producing token ids bit-identical to the Evaluator path.

  `native_fallback: :raise` makes this a no-fallback gate: every op in the
  whole generation graph — the transformer forward, the `while` loop, the
  dynamic KV-cache writes, `cumsum` for position ids, `argmax`/sampling for
  token selection, threefry for the sampling seed — must lower, or the build
  raises rather than silently degrading to the evaluator.

  Driven through `build_generate` on small in-vocabulary `input_ids` (not
  text) so the test needs no tokenizer and stays clear of the OOB-gather
  pitfall of pairing a tiny-random model with a full-vocab tokenizer.
  """
  use ExUnit.Case, async: false
  @moduletag :conformance

  alias Bumblebee.Text.Generation

  @repo {:hf, "bumblebee-testing/tiny-random-Qwen3ForCausalLM"}

  setup_all do
    Nx.global_default_backend(Emily.Backend)
    {:ok, model_info} = Bumblebee.load_model(@repo)
    {:ok, gen_config} = Bumblebee.load_generation_config(@repo)
    %{model_info: model_info, gen_config: gen_config}
  end

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]
  # CM14: the fused-while lane — same no-fallback native compile, but the
  # decode loop's body is fused under `mx::compile`. The fusion reassociates
  # f32, so logits are not bit-identical; greedy argmax over them is, so the
  # acceptance gate is a token-id match rather than a binary-identical one.
  @native_compiled [
    compiler: Emily.Compiler,
    native: true,
    native_fallback: :raise,
    native_compiled: true
  ]

  # Run `model`'s generation through `build_generate` on a fixed in-vocab
  # prompt, returning the generated token ids.
  defp generate_ids(%{model: model, params: params, spec: spec}, gen_config, defn_options) do
    generate = Generation.build_generate(model, spec, gen_config)

    inputs = %{
      "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6]], type: :s64, backend: Emily.Backend),
      "seed" => Nx.tensor([0], type: :s64, backend: Emily.Backend)
    }

    Nx.Defn.jit(generate, defn_options).(params, inputs).token_ids
    |> Nx.to_flat_list()
  end

  defp configure(gen_config, strategy) do
    Bumblebee.configure(gen_config,
      max_new_tokens: 6,
      strategy: strategy,
      pad_token_id: 0,
      eos_token_id: 0
    )
  end

  test "greedy generation compiles fully native, bit-identical to the evaluator", ctx do
    gc = configure(ctx.gen_config, %{type: :greedy_search})
    native = generate_ids(ctx.model_info, gc, @native)
    eval = generate_ids(ctx.model_info, gc, @eval)
    assert native == eval
  end

  test "multinomial sampling compiles fully native, bit-identical to the evaluator", ctx do
    gc = configure(ctx.gen_config, %{type: :multinomial_sampling})
    native = generate_ids(ctx.model_info, gc, @native)
    eval = generate_ids(ctx.model_info, gc, @eval)
    assert native == eval
  end

  test "greedy generation with the fused-while body matches the evaluator's token ids", ctx do
    # The opt-in fused-while lane (CM14): the `defn while` decode loop runs
    # host-controlled, but each per-token forward (the loop body) replays
    # through a per-stream-cached `mx::compile`'d callable. `mx::compile`
    # reassociates f32 so the logits drift by a few ULP — greedy argmax is
    # robust to that drift, so the token ids match here. This is a
    # token-id gate, not a binary-identical one (the logits aren't), and
    # the match is empirical: a near-tie top-2 logit could in principle
    # flip a token on another model/prompt. Sampling strategies would
    # diverge under fusion, so only greedy is gated.
    gc = configure(ctx.gen_config, %{type: :greedy_search})
    fused = generate_ids(ctx.model_info, gc, @native_compiled)
    eval = generate_ids(ctx.model_info, gc, @eval)
    assert fused == eval
  end
end
