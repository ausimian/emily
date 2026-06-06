# Qwen3-0.6B greedy-decode throughput on `Emily.Backend`.
#
# Reports two lanes by default and prints the speedup between them:
#
#   * baseline — the op-by-op `Nx.Defn.Evaluator` walk.
#   * native   — the single-NIF `Emily.Compiler` (`native: true`): the whole
#                generation graph, including the `defn while` decode loop,
#                replays as one NIF call per step. Run with
#                `native_fallback: :raise`, so the number is a true
#                full-native measurement, not a silent fallback.
#
# (The opt-in `EMILY_BENCH_FAST_KERNELS` lane is orthogonal — it benches the
# fused MLX kernels under the evaluator.)
#
# Usage:
#
#     elixir bench/qwen3_tokens_per_sec.exs
#
# Standalone via Mix.install so a reader can run it without the project's
# test setup; the dep versions below track Emily's own (keep in sync).
#
# Optional environment variables:
#
#     EMILY_BENCH_MODEL          HuggingFace repo id. Defaults to
#                                "Qwen/Qwen3-0.6B".
#     EMILY_BENCH_NEW_TOKENS     Number of tokens to greedy-decode per
#                                run. Defaults to 64.
#     EMILY_BENCH_PROMPT         Prompt text. Defaults to a fixed short
#                                English sentence.
#     EMILY_BENCH_WARMUP         Number of warm-up runs (not measured).
#                                Defaults to 1.
#     EMILY_BENCH_RUNS           Number of measured runs. Defaults to 3.
#     EMILY_BENCH_FAST_KERNELS   "1" → also benchmark the M11 fused
#                                MLX kernels (RMSNorm, LayerNorm, RoPE,
#                                SDPA) via `Emily.Bumblebee.FastKernels`.
#                                Reports baseline vs fused side by side.
#                                Requires `Emily.Bumblebee.FastKernels`
#                                to be available in `lib/` (see the
#                                separate graduation PR); script raises
#                                if it isn't.
#     EMILY_BENCH_PIN            "1.5" → fail with non-zero exit if the
#                                fused mean tokens/sec doesn't beat
#                                baseline mean by at least the given
#                                multiplier. Implies
#                                EMILY_BENCH_FAST_KERNELS=1.
#
# The first run downloads the model (~1.2 GB at f32, ~600 MB at f16).
# We deliberately avoid `Benchee` — this benchmark has one workload and
# one (or two) metrics. The whole script is standalone so a reader can
# follow the generation flow without chasing macros.

Mix.install([
  {:emily, path: Path.expand("..", __DIR__)},
  # Versions track Emily's own deps (Bumblebee 0.7 / Axon 0.8 / Nx 0.12);
  # keep them in sync with mix.exs so the standalone install resolves.
  {:bumblebee, "~> 0.7"},
  {:axon, "~> 0.8"},
  {:tokenizers, "~> 0.5"},
  {:nx, "~> 0.12"}
])

defmodule Emily.Bench.Qwen3 do
  @default_model "Qwen/Qwen3-0.6B"
  @default_prompt "The quick brown fox jumps over the lazy dog."
  @default_new_tokens 64
  @default_warmup 1
  @default_runs 3

  def run do
    Nx.global_default_backend(Emily.Backend)

    model_repo = System.get_env("EMILY_BENCH_MODEL", @default_model)
    prompt = System.get_env("EMILY_BENCH_PROMPT", @default_prompt)

    new_tokens = System.get_env("EMILY_BENCH_NEW_TOKENS") |> env_int(@default_new_tokens)
    warmup = System.get_env("EMILY_BENCH_WARMUP") |> env_int(@default_warmup)
    runs = System.get_env("EMILY_BENCH_RUNS") |> env_int(@default_runs)

    pin_threshold =
      case System.get_env("EMILY_BENCH_PIN") do
        nil -> nil
        s -> elem(Float.parse(s), 0)
      end

    fast_kernels? =
      System.get_env("EMILY_BENCH_FAST_KERNELS") == "1" or pin_threshold != nil

    IO.puts("Emily / Qwen3 throughput benchmark")
    IO.puts("  model          : #{model_repo}")
    IO.puts("  prompt         : #{inspect(prompt)}")
    IO.puts("  new tokens     : #{new_tokens}")
    IO.puts("  warmup         : #{warmup}")
    IO.puts("  runs           : #{runs}")
    IO.puts("  fused kernels  : #{fast_kernels?}")
    if pin_threshold, do: IO.puts("  pin threshold  : #{pin_threshold}× baseline")
    IO.puts("")

    {:ok, model_info} = Bumblebee.load_model({:hf, model_repo})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_repo})

    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, model_repo})

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: new_tokens,
        strategy: %{type: :greedy_search}
      )

    cfg = %{
      tokenizer: tokenizer,
      generation_config: generation_config,
      prompt: prompt,
      new_tokens: new_tokens,
      warmup: warmup,
      runs: runs
    }

    # Baseline: the op-by-op Evaluator walk (~one BEAM↔worker round-trip per
    # op). Native: the single-NIF replay — the whole generation graph,
    # including the `defn while` decode loop, runs as one NIF call per step
    # (`native_fallback: :raise` so this is a true full-native measurement,
    # never a silent fallback to the evaluator).
    {baseline_mean, _, _, _} =
      bench("baseline (evaluator, op-by-op)", model_info, [compiler: Nx.Defn.Evaluator], cfg)

    {native_mean, _, _, _} =
      bench(
        "native (single-NIF Emily.Compiler)",
        model_info,
        [compiler: Emily.Compiler, native: true, native_fallback: :raise],
        cfg
      )

    IO.puts("\nnative speedup : #{Float.round(native_mean / baseline_mean, 2)}× over the evaluator")

    if fast_kernels? do
      unless Code.ensure_loaded?(Emily.Bumblebee.FastKernels) do
        IO.puts(
          "EMILY_BENCH_FAST_KERNELS=1 requires Emily.Bumblebee.FastKernels to be\n" <>
            "in `lib/`, which isn't the case on this branch. Graduate the shim first."
        )

        System.halt(1)
      end

      fused_model_info = update_in(model_info.model, &Emily.Bumblebee.FastKernels.apply/1)

      {fused_mean, _, _, _} =
        bench(
          "fused (Emily.Bumblebee.FastKernels)",
          fused_model_info,
          [compiler: Nx.Defn.Evaluator],
          cfg
        )

      speedup = fused_mean / baseline_mean
      IO.puts("\nfused speedup  : #{Float.round(speedup, 2)}× (fused mean / baseline mean)")

      if pin_threshold do
        if speedup >= pin_threshold do
          IO.puts("PIN OK         : #{Float.round(speedup, 2)}× ≥ #{pin_threshold}×")
        else
          IO.puts("PIN FAIL       : #{Float.round(speedup, 2)}× < #{pin_threshold}×")
          System.halt(1)
        end
      end
    end
  end

  defp bench(label, model_info, defn_options, cfg) do
    %{
      tokenizer: tokenizer,
      generation_config: generation_config,
      prompt: prompt,
      new_tokens: new_tokens,
      warmup: warmup,
      runs: runs
    } = cfg

    IO.puts("=== #{label} ===")

    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        defn_options: defn_options
      )

    for _ <- Stream.duplicate(:ok, warmup) do
      IO.puts("[warmup] generating…")
      %{results: [_]} = Nx.Serving.run(serving, prompt)
    end

    measurements =
      for n <- 1..runs//1 do
        {elapsed_us, %{results: [%{text: text}]}} =
          :timer.tc(fn -> Nx.Serving.run(serving, prompt) end)

        secs = elapsed_us / 1_000_000
        tps = new_tokens / secs
        IO.puts("[run #{n}] #{Float.round(secs, 3)} s, #{Float.round(tps, 2)} tok/s")
        {secs, tps, text}
      end

    tps_list = Enum.map(measurements, fn {_, tps, _} -> tps end)
    [{_, _, sample} | _] = measurements
    mean = Enum.sum(tps_list) / length(tps_list)
    {min_tps, max_tps} = Enum.min_max(tps_list)

    IO.puts(
      "tokens/sec     : mean=#{Float.round(mean, 2)}  min=#{Float.round(min_tps, 2)}  max=#{Float.round(max_tps, 2)}"
    )

    IO.puts("first completion:\n  #{String.slice(sample, 0, 500)}")
    {mean, min_tps, max_tps, sample}
  end

  defp env_int(nil, default), do: default

  defp env_int(s, default) do
    case Integer.parse(s) do
      {n, ""} -> n
      _ -> default
    end
  end
end

Emily.Bench.Qwen3.run()
