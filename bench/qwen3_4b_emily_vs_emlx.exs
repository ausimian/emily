# Qwen3-4B greedy-decode throughput: Emily vs EMLX vs EXLA.
#
# This is intentionally close to `bench/qwen3_tokens_per_sec.exs`, but it
# compares Emily's execution lanes against EMLX and EXLA on the same Bumblebee
# model.
#
# Lanes:
#
#   * exla         - EXLA compiler on EXLA.Backend (host CPU on Apple Silicon).
#   * emlx         - EMLX compiler on EMLX.Backend (Metal GPU).
#   * emily-eager  - Nx.Defn.Evaluator on Emily.Backend, the op-by-op baseline.
#   * emily-native - Emily.Compiler with `native: true`, replaying the decode
#                    graph through one NIF call per step.
#   * emily-fuse   - emily-native plus `fuse: true`, caching an `mx::compile`'d
#                    decode loop body per stream.
#
# Usage:
#
#     elixir bench/qwen3_4b_emily_vs_emlx.exs
#
# The default 4B run is GPU-only (`emlx,emily-eager,emily-native,emily-fuse`).
# The EXLA lane is implemented for explicit experiments, but Qwen3-4B bf16 on
# EXLA-CPU was killed by the OS on a 24 GB M4 Pro during compile/run. For the
# canonical three-way benchmark that completes on this machine, use
# `bench/emily_vs_exla.exs`.
#
# Optional environment variables:
#
#     EMILY_BENCH_MODEL       HuggingFace repo id. Defaults to "Qwen/Qwen3-4B".
#                             `Qwen/Qwen3-8B` is the largest plausible opt-in
#                             target for a 24 GB M4 MBP.
#     EMILY_BENCH_TYPE        Parameter/compute type. Defaults to "bf16".
#                             Accepted: "bf16", "f16", "f32".
#     EMILY_BENCH_NEW_TOKENS  Number of tokens to greedy-decode per run.
#                             Defaults to 32.
#     EMILY_BENCH_PROMPT      Prompt text. Defaults to a fixed short sentence.
#     EMILY_BENCH_WARMUP      Number of warm-up runs per lane. Defaults to 1.
#     EMILY_BENCH_RUNS        Number of measured runs per lane. Defaults to 3.
#     EMILY_BENCH_LANES       Comma-separated subset of:
#                             exla,emlx,emily-eager,emily-native,emily-fuse.
#
# The first run downloads Qwen3-4B (~8 GB in bf16). The script loads the model
# separately for EXLA, EMLX, and Emily so each backend owns its own parameter
# tensors; load time is printed but not included in tokens/sec.

Mix.install([
  {:emily, path: Path.expand("..", __DIR__)},
  {:exla, "~> 0.12"},
  {:emlx, "~> 0.3.1"},
  # Versions track Emily's own deps (Bumblebee 0.7 / Axon 0.8 / Nx 0.12);
  # keep them in sync with mix.exs so the standalone install resolves.
  {:bumblebee, "~> 0.7"},
  {:axon, "~> 0.8"},
  {:tokenizers, "~> 0.5"},
  {:nx, "~> 0.12"}
])

defmodule Emily.Bench.Qwen3ThreeWay do
  @default_model "Qwen/Qwen3-4B"
  @default_prompt "The quick brown fox jumps over the lazy dog."
  @default_type :bf16
  @default_new_tokens 32
  @default_warmup 1
  @default_runs 3
  @default_lane_ids ~w(emlx emily-eager emily-native emily-fuse)

  @all_lanes [
    %{
      id: "exla",
      label: "exla (EXLA compiler + EXLA.Backend / CPU)",
      backend: EXLA.Backend,
      defn_options: [compiler: EXLA],
      owner: :exla
    },
    %{
      id: "emlx",
      label: "emlx (EMLX compiler + EMLX.Backend)",
      backend: {EMLX.Backend, device: :gpu},
      defn_options: [compiler: EMLX],
      owner: :emlx
    },
    %{
      id: "emily-eager",
      label: "emily-eager (Evaluator + Emily.Backend)",
      backend: Emily.Backend,
      defn_options: [compiler: Nx.Defn.Evaluator],
      owner: :emily
    },
    %{
      id: "emily-native",
      label: "emily-native (Emily.Compiler native)",
      backend: Emily.Backend,
      defn_options: [compiler: Emily.Compiler, native: true, native_fallback: :raise],
      owner: :emily
    },
    %{
      id: "emily-fuse",
      label: "emily-fuse (native + mx::compile'd loop body)",
      backend: Emily.Backend,
      defn_options: [
        compiler: Emily.Compiler,
        native: true,
        native_fallback: :raise,
        fuse: true
      ],
      owner: :emily
    }
  ]

  def run do
    model_repo = System.get_env("EMILY_BENCH_MODEL", @default_model)
    prompt = System.get_env("EMILY_BENCH_PROMPT", @default_prompt)
    type = System.get_env("EMILY_BENCH_TYPE") |> env_type(@default_type)
    new_tokens = System.get_env("EMILY_BENCH_NEW_TOKENS") |> env_int(@default_new_tokens)
    warmup = System.get_env("EMILY_BENCH_WARMUP") |> env_non_neg_int(@default_warmup)
    runs = System.get_env("EMILY_BENCH_RUNS") |> env_int(@default_runs)
    lanes = selected_lanes(System.get_env("EMILY_BENCH_LANES"))

    IO.puts("Emily vs EMLX vs EXLA / Qwen3 throughput benchmark")
    IO.puts("  model      : #{model_repo}")
    IO.puts("  type       : #{inspect(type)}")
    IO.puts("  prompt     : #{inspect(prompt)}")
    IO.puts("  new tokens : #{new_tokens}")
    IO.puts("  warmup     : #{warmup}")
    IO.puts("  runs       : #{runs}")
    IO.puts("  lanes      : #{Enum.map_join(lanes, ", ", & &1.id)}")
    IO.puts("")

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

    results =
      lanes
      |> Enum.chunk_by(& &1.owner)
      |> Enum.flat_map(&bench_group(&1, model_repo, type, cfg))

    print_summary(results)
  end

  defp bench_group(owner_lanes, model_repo, type, cfg) do
    backend =
      owner_lanes
      |> List.first()
      |> Map.fetch!(:backend)

    safe(fn ->
      with_model(backend, model_repo, type, fn model_info ->
        Enum.map(owner_lanes, &bench_lane(&1, model_info, cfg))
      end)
    end)
    |> case do
      {:ok, results} ->
        results

      {:error, error} ->
        Enum.map(owner_lanes, &%{lane: &1.id, error: error})
    end
  end

  defp with_model(backend, model_repo, type, fun) do
    reclaim()
    prev_backend = Nx.global_default_backend(backend)

    try do
      IO.puts("loading #{model_repo} on #{inspect(backend)} as #{inspect(type)}...")

      {load_us, {:ok, model_info}} =
        :timer.tc(fn -> Bumblebee.load_model({:hf, model_repo}, type: type) end)

      IO.puts("loaded in #{Float.round(load_us / 1_000_000, 2)} s")
      fun.(model_info)
    after
      Nx.global_default_backend(prev_backend)
      reclaim()
    end
  end

  defp bench_lane(lane, model_info, cfg) do
    %{
      tokenizer: tokenizer,
      generation_config: generation_config,
      prompt: prompt,
      new_tokens: new_tokens,
      warmup: warmup,
      runs: runs
    } = cfg

    IO.puts("\n=== #{lane.label} ===")
    prev_backend = Nx.global_default_backend(lane.backend)

    try do
      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config,
          defn_options: lane.defn_options
        )

      for _ <- Stream.duplicate(:ok, warmup) do
        IO.puts("[warmup] generating...")
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
        "tokens/sec: mean=#{Float.round(mean, 2)}  min=#{Float.round(min_tps, 2)}  max=#{Float.round(max_tps, 2)}"
      )

      IO.puts("first completion:\n  #{String.slice(sample, 0, 500)}")

      %{lane: lane.id, mean: mean, min: min_tps, max: max_tps, sample: sample}
    rescue
      e ->
        msg = Exception.format(:error, e, __STACKTRACE__)
        IO.puts("ERROR:\n#{msg}")
        %{lane: lane.id, error: Exception.message(e)}
    after
      Nx.global_default_backend(prev_backend)
    end
  end

  defp print_summary(results) do
    exla_mean =
      Enum.find_value(results, fn
        %{lane: "exla", mean: mean} -> mean
        _ -> nil
      end)

    emlx_mean =
      Enum.find_value(results, fn
        %{lane: "emlx", mean: mean} -> mean
        _ -> nil
      end)

    IO.puts("\nSummary")
    IO.puts("| lane | mean tok/s | min | max | speedup vs exla | speedup vs emlx |")
    IO.puts("| ---- | ----------: | --: | --: | --------------: | --------------: |")

    for result <- results do
      case result do
        %{lane: lane, mean: mean, min: min_tps, max: max_tps} ->
          exla_speedup = speedup(mean, exla_mean)
          emlx_speedup = speedup(mean, emlx_mean)

          IO.puts(
            "| #{lane} | #{Float.round(mean, 2)} | #{Float.round(min_tps, 2)} | #{Float.round(max_tps, 2)} | #{exla_speedup} | #{emlx_speedup} |"
          )

        %{lane: lane, error: error} ->
          IO.puts("| #{lane} | ERROR: #{String.replace(error, "|", "\\|")} | - | - | - | - |")
      end
    end

    best =
      results
      |> Enum.filter(&match?(%{mean: _}, &1))
      |> Enum.filter(&String.starts_with?(&1.lane, "emily-"))
      |> Enum.max_by(& &1.mean, fn -> nil end)

    if best do
      IO.puts("\nBest Emily lane: #{best.lane} at #{Float.round(best.mean, 2)} tok/s")

      if exla_mean do
        IO.puts("Best Emily speedup vs EXLA: #{speedup(best.mean, exla_mean)}")
      end

      if emlx_mean do
        IO.puts("Best Emily speedup vs EMLX: #{speedup(best.mean, emlx_mean)}")
      end
    end
  end

  defp speedup(_mean, nil), do: "-"
  defp speedup(mean, baseline), do: "#{Float.round(mean / baseline, 2)}x"

  defp selected_lanes(nil) do
    ids = MapSet.new(@default_lane_ids)
    Enum.filter(@all_lanes, &MapSet.member?(ids, &1.id))
  end

  defp selected_lanes(value) do
    ids =
      value
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)
      |> MapSet.new()

    @all_lanes
    |> Enum.filter(&MapSet.member?(ids, &1.id))
    |> case do
      [] -> raise ArgumentError, "EMILY_BENCH_LANES did not match any known lane"
      lanes -> lanes
    end
  end

  defp env_int(nil, default), do: default

  defp env_int(s, default) do
    case Integer.parse(s) do
      {n, ""} when n > 0 -> n
      _ -> default
    end
  end

  defp env_non_neg_int(nil, default), do: default

  defp env_non_neg_int(s, default) do
    case Integer.parse(s) do
      {n, ""} when n >= 0 -> n
      _ -> default
    end
  end

  defp env_type(nil, default), do: default
  defp env_type("bf16", _default), do: :bf16
  defp env_type("f16", _default), do: :f16
  defp env_type("f32", _default), do: :f32
  defp env_type(_other, default), do: default

  defp safe(fun) do
    {:ok, fun.()}
  rescue
    e -> {:error, Exception.message(e)}
  catch
    kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
  end

  defp reclaim do
    :erlang.garbage_collect()

    if Code.ensure_loaded?(Emily.Memory) and function_exported?(Emily.Memory, :clear_cache, 0) do
      Emily.Memory.clear_cache()
    end

    if Code.ensure_loaded?(EMLX) and function_exported?(EMLX, :clear_cache, 0) do
      EMLX.clear_cache()
    end

    :erlang.garbage_collect()
    :ok
  end
end

Emily.Bench.Qwen3ThreeWay.run()
