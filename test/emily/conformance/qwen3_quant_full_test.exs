defmodule Emily.Conformance.Qwen3QuantFullTest do
  @moduledoc """
  End-to-end quantized-inference conformance for `Qwen/Qwen3-0.6B`.

  Loads dense Qwen3-0.6B via Bumblebee, quantizes via
  `Emily.Quantization.Transform.quantize/3` (bits=4, group_size=128,
  transpose=true), greedy-decodes 32 tokens, gates the completion
  against a pinned reference.

  Excluded even from `--only conformance` — ~1.2 GB model download
  plus full quantization pass. Run explicitly:

      mix test --only qwen3_quant_full

  A pinned-text drift flags: defn-dequantize bugs, Transform rewriter
  drift, Bumblebee Qwen3-port changes, or MLX quantize/dequantize
  changes. (Real AWQ-safetensors loading is deferred to a follow-up;
  see PLAN.md M10.5.)
  """

  use ExUnit.Case, async: true

  alias Emily.Bumblebee.FastKernels
  alias Emily.Quantization.Transform

  @moduletag :qwen3_quant_full
  @moduletag capture_log: true
  @moduletag timeout: 1_200_000

  @prompt "The quick brown fox jumps over the lazy dog."

  # Reference produced by running this same pipeline on a clean
  # Emily.Backend checkout (M10.5 first run). A drift in this string
  # means something in the quantization stack has changed; investigate
  # before bumping. Note the output is coherent English but differs
  # from the dense Qwen3-0.6B reference — expected given int4
  # quantization noise across all linear layers.
  @reference_text " Let's see, what is the correct answer for this riddle? The answer is a word that contains the letters B, O, U, and R,"

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  test "Qwen/Qwen3-0.6B quantized greedy decode matches the pinned continuation" do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-0.6B"})

    # Quantize the model: rewrite :dense nodes to :quantized_dense and
    # replace every dense kernel in params with %QuantizedWeight{}.
    # transpose=true groups along the reduction axis (the LLM-accuracy
    # convention).
    {qmodel, qparams} =
      Transform.quantize(model, params,
        bits: 4,
        group_size: 128,
        transpose: true
      )

    model_info = %{model: qmodel, params: qparams, spec: spec}

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
  test "Qwen/Qwen3-0.6B quantized + fused MLX kernels still decodes 32 tokens" do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-0.6B"})

    # Compose the two rewrites: quantize first (rewrites dense → quantized_dense
    # by name; doesn't touch RMSNorm / RoPE / attention nodes) then fuse
    # (rewrites the still-stock norm/rope/attention nodes). Order matters
    # only when the two rewrites would compete on the same nodes — they
    # don't here.
    {qmodel, qparams} =
      Transform.quantize(model, params,
        bits: 4,
        group_size: 128,
        transpose: true
      )

    fast_qmodel = FastKernels.apply(qmodel)

    model_info = %{model: fast_qmodel, params: qparams, spec: spec}

    config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 32,
        strategy: %{type: :greedy_search}
      )

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, config,
        defn_options: [compiler: Nx.Defn.Evaluator]
      )

    %{results: [%{token_summary: summary}]} =
      Nx.Serving.run(serving, @prompt)

    # Both quantization noise *and* fused-kernel reordering operate on
    # the logits — combined drift makes a pinned-text assertion brittle.
    # The structural assertion (32 tokens out, no crash) is the
    # informative one for this milestone.
    assert summary.output == 32
  end
end
