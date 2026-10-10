defmodule Emily.Conformance.Qwen3FullTest do
  @moduledoc """
  The real `Qwen/Qwen3-0.6B` decoder on Emily's compiler lanes.

  Stock Bumblebee on `Emily.Backend` (evaluator) is the oracle. Every
  lane must reproduce its prefill logits and its greedy token ids. Decode
  runs through Bumblebee's own `defn while` loop (`build_generate`) under
  `native_fallback: :raise`, so the whole generation graph must lower
  natively. The batched case pads the shorter prompt on the left, so
  each row decodes from its own position offset.

  The lanes:

    * The stock model, native and with `fuse: true` (the loop body fused
      under `mx::compile`). `generation_native_test.exs` gates both on a
      tiny-random Qwen3; here they meet a real vocabulary, real weights
      and real shapes.
    * `Emily.Bumblebee.FastKernels` on the evaluator and native. ViT,
      Whisper and DistilBERT use LayerNorm and absolute positions, so
      the fused RMSNorm and RoPE kernels, causal grouped-query attention
      and KV-cache decode are covered here. Qwen3-0.6B has 28 blocks,
      each with RMSNorm (including per-head query and key norms), RoPE
      and grouped-query attention (16 query heads over 8 key/value
      heads, head dim 128).
    * `Bumblebee.Text.generation` through `Nx.Serving`, which tokenizes,
      compiles for a fixed batch size and sequence length, and
      detokenizes.

  Excluded by default (~1.5 GB checkpoint). Run with:

      mix test --only qwen3_full
  """

  use ExUnit.Case, async: false
  use Emily.ConformanceHelper

  alias Bumblebee.Text.Generation
  alias Emily.Bumblebee.FastKernels
  alias Emily.FastKernelsRewrites

  @moduletag :qwen3_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  @repo {:hf, "Qwen/Qwen3-0.6B"}

  @eval [compiler: Emily.Compiler, native: false]
  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @fuse @native ++ [fuse: true]

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

    gen_config =
      Bumblebee.configure(gen_config, max_new_tokens: 24, strategy: %{type: :greedy_search})

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

  # Each variant is {name, `FastKernels.apply/2` options (`:stock` for the
  # unmodified model), defn options, prefill logits check}.
  for {variant, kernel_opts, predict_opts, logits_check} <- [
        {"stock model (native)", :stock, @native, :exact},
        {"stock model (fuse)", :stock, @fuse, :close},
        {"fused kernels (evaluator)", [], @eval, :close},
        {"fused kernels (native)", [], @native, :close},
        {"fused kernels (native, force_fused_sdpa)", [force_fused_sdpa: true], @native, :close}
      ] do
    describe variant do
      setup ctx do
        %{model: variant_model(ctx.model_info.model, unquote(kernel_opts))}
      end

      test "prefill logits match stock", ctx do
        logits =
          Axon.predict(ctx.model, ctx.model_info.params, ctx.inputs, unquote(predict_opts)).logits

        assert_logits(unquote(logits_check), logits, ctx.stock_logits)
      end

      test "greedy decode matches stock token ids", ctx do
        assert generate(ctx, ctx.model, ctx.inputs, unquote(predict_opts)) == ctx.stock_ids
      end

      test "left-padded batched greedy decode matches stock token ids", ctx do
        assert generate(ctx, ctx.model, ctx.batch_inputs, unquote(predict_opts)) ==
                 ctx.stock_batch_ids
      end
    end
  end

  describe "Nx.Serving" do
    test "native text generation matches the evaluator", ctx do
      eval = serve(ctx, @eval)

      assert [%{results: [%{text: text}]}, _] = eval
      assert text =~ ~r/^ Paris\b/

      assert serve(ctx, @native) == eval
    end
  end

  defp variant_model(model, :stock), do: model
  defp variant_model(model, kernel_opts), do: FastKernels.apply(model, kernel_opts)

  # The native replay runs the evaluator's MLX ops one for one, so its
  # logits are bit-identical.
  defp assert_logits(:exact, logits, stock_logits) do
    assert Nx.to_binary(logits) == Nx.to_binary(stock_logits)
  end

  # Fused kernels and `mx::compile` reassociate f32. Observed max |diff|
  # ~4e-5 against a top-2 logit gap of ~3.
  defp assert_logits(:close, logits, stock_logits) do
    assert_all_close(logits, stock_logits, atol: 1.0e-3, rtol: 1.0e-3)

    assert Nx.to_number(Nx.argmax(logits[[0, -1]])) ==
             Nx.to_number(Nx.argmax(stock_logits[[0, -1]]))
  end

  # Greedy-decode `inputs` with `model` through Bumblebee's own `defn while`
  # loop, returning each row's generated token ids.
  defp generate(ctx, model, inputs, defn_options) do
    generate = Generation.build_generate(model, ctx.model_info.spec, ctx.gen_config)
    batch = Nx.axis_size(inputs["input_ids"], 0)
    inputs = Map.put(inputs, "seed", Nx.broadcast(0, {batch}))

    Nx.Defn.jit(generate, defn_options).(ctx.model_info.params, inputs).token_ids
    |> Nx.to_list()
  end

  # Generate text for both batch prompts through `Bumblebee.Text.generation`,
  # compiled for a fixed shape the way a deployed serving would be.
  defp serve(ctx, defn_options) do
    ctx.model_info
    |> Bumblebee.Text.generation(ctx.tokenizer, ctx.gen_config,
      compile: [batch_size: 2, sequence_length: 32],
      defn_options: defn_options
    )
    |> Nx.Serving.run(@batch)
  end
end
