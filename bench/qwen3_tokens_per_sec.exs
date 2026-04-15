# Qwen3-0.6B greedy-decode throughput on `Emily.Backend`.
#
# Usage:
#
#     MIX_ENV=test mix run bench/qwen3_tokens_per_sec.exs
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

    IO.puts("=== baseline (composed defn kernels) ===")
    baseline = bench(model_info, tokenizer, generation_config, prompt, new_tokens, warmup, runs)
    {baseline_mean, _, _, _} = baseline

    if fast_kernels? do
      IO.puts("\n=== fused (Emily.Bumblebee.FastKernels) ===")

      fused_model_info =
        update_in(model_info.model, &Emily.Bumblebee.FastKernels.apply/1)

      fused = bench(fused_model_info, tokenizer, generation_config, prompt, new_tokens, warmup, runs)
      {fused_mean, _, _, _} = fused

      speedup = fused_mean / baseline_mean
      IO.puts("\nspeedup        : #{Float.round(speedup, 2)}× (fused mean / baseline mean)")

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

  defp bench(model_info, tokenizer, generation_config, prompt, new_tokens, warmup, runs) do
    serving =
      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        defn_options: [compiler: Nx.Defn.Evaluator]
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
