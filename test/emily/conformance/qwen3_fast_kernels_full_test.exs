defmodule Emily.Conformance.Qwen3FastKernelsFullTest do
  @moduledoc """
  `Emily.Bumblebee.FastKernels` on the real `Qwen/Qwen3-0.6B` decoder.

  ViT, Whisper and DistilBERT use LayerNorm and absolute positions, so
  the fused RMSNorm and RoPE kernels, causal grouped-query attention and
  KV-cache decode are covered here. Qwen3-0.6B has 28 blocks, each with
  RMSNorm (including per-head query and key norms), RoPE and
  grouped-query attention (16 query heads over 8 key/value heads, head
  dim 128).

  Stock Bumblebee on `Emily.Backend` (evaluator) is the oracle; every
  fused variant must reproduce its prefill logits and its greedy token
  ids. The batched case pads the shorter prompt on the left, so each row
  decodes from its own position offset.

  Excluded by default (~1.5 GB checkpoint). Run with:

      mix test --only fast_kernels_full
  """

  use ExUnit.Case, async: false
  use Emily.ConformanceHelper

  alias Bumblebee.Text.Generation
  alias Emily.Bumblebee.FastKernels
  alias Emily.FastKernelsRewrites

  @moduletag :fast_kernels_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  @repo {:hf, "Qwen/Qwen3-0.6B"}

  @eval [compiler: Emily.Compiler, native: false]
  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]

  @prompt "The capital of France is"
  @batch [@prompt, "Here is a short poem about the sea, written in the style of a sailor:"]

  setup_all do
    # `setup_all` runs in its own process, before `ConformanceHelper`'s
    # per-test `setup` sets the default backend.
    Nx.default_backend(Emily.Backend)
    {:ok, model_info} = Bumblebee.load_model(@repo)
    {:ok, tokenizer} = Bumblebee.load_tokenizer(@repo)
    {:ok, gen_config} = Bumblebee.load_generation_config(@repo)

    tokenizer = Bumblebee.configure(tokenizer, pad_direction: :left)

    ctx = %{
      model_info: model_info,
      tokenizer: tokenizer,
      gen_config: gen_config,
      inputs: Bumblebee.apply_tokenizer(tokenizer, @prompt),
      batch_inputs: Bumblebee.apply_tokenizer(tokenizer, @batch)
    }

    stock = model_info.model

    Map.merge(ctx, %{
      stock_logits: Axon.predict(stock, model_info.params, ctx.inputs, @eval).logits,
      stock_ids: generate(ctx, stock, ctx.inputs, @eval),
      stock_batch_ids: generate(ctx, stock, ctx.batch_inputs, @eval)
    })
  end

  test "FastKernels rewrites every RMSNorm, RoPE and attention layer", ctx do
    blocks = ctx.model_info.spec.num_blocks

    # Per block: input and post-attention norms, query and key norms, one
    # RoPE and one attention; plus the final norm.
    assert FastKernelsRewrites.counts(FastKernels.apply(ctx.model_info.model)) ==
             %{
               fast_rms_norm_impl: 4 * blocks + 1,
               fast_rope_impl: blocks,
               fast_sdpa_impl: blocks
             }
  end

  test "stock greedy decode is coherent", ctx do
    # Guards the oracle itself: a broken stock path would make every
    # comparison below vacuous.
    assert Bumblebee.Tokenizer.decode(ctx.tokenizer, hd(ctx.stock_ids)) =~ ~r/^ Paris\b/
  end

  for {variant, apply_opts, predict_opts} <- [
        {"evaluator", [], @eval},
        {"native", [], @native},
        {"native, force_fused_sdpa", [force_fused_sdpa: true], @native}
      ] do
    describe "fused kernels (#{variant})" do
      setup ctx do
        %{fast: FastKernels.apply(ctx.model_info.model, unquote(apply_opts))}
      end

      test "prefill logits match stock", ctx do
        logits =
          Axon.predict(ctx.fast, ctx.model_info.params, ctx.inputs, unquote(predict_opts)).logits

        # Observed max |diff| ~4e-5 against a top-2 logit gap of ~3.
        assert_all_close(logits, ctx.stock_logits, atol: 1.0e-3, rtol: 1.0e-3)

        assert Nx.to_number(Nx.argmax(logits[[0, -1]])) ==
                 Nx.to_number(Nx.argmax(ctx.stock_logits[[0, -1]]))
      end

      test "greedy decode matches stock token ids", ctx do
        assert generate(ctx, ctx.fast, ctx.inputs, unquote(predict_opts)) == ctx.stock_ids
      end

      test "left-padded batched greedy decode matches stock token ids", ctx do
        assert generate(ctx, ctx.fast, ctx.batch_inputs, unquote(predict_opts)) ==
                 ctx.stock_batch_ids
      end
    end
  end

  # Greedy-decode `inputs` with `model` through Bumblebee's own `defn while`
  # loop, returning each row's generated token ids.
  defp generate(ctx, model, inputs, defn_options) do
    gen_config =
      Bumblebee.configure(ctx.gen_config,
        max_new_tokens: 24,
        strategy: %{type: :greedy_search}
      )

    generate = Generation.build_generate(model, ctx.model_info.spec, gen_config)
    batch = Nx.axis_size(inputs["input_ids"], 0)
    inputs = Map.put(inputs, "seed", Nx.broadcast(0, {batch}))

    Nx.Defn.jit(generate, defn_options).(ctx.model_info.params, inputs).token_ids
    |> Nx.to_list()
  end
end
