# End-to-end quantized Qwen3-0.6B greedy-decode throughput on Emily's
# native lane. Loads the dense model, rewrites every dense layer to
# `Emily.Quantization.Layers.quantized_dense/4` + quantizes the params
# (affine 4-bit) via `Emily.Quantization.Transform`, then measures
# tokens/sec through a compiled `Bumblebee.Text.generation` serving.
#
# Run in TEST env so `Emily.Quantization.Transform` (test/support) is on
# the compile path, and in-project (NOT Mix.install) so it exercises the
# LOCAL build:
#
#     MIX_ENV=test mix run bench/qwen3_quantized_tps.exs
#
# Optional env: EMILY_BENCH_MODEL, EMILY_BENCH_NEW_TOKENS (64),
# EMILY_BENCH_WARMUP (1), EMILY_BENCH_RUNS (3), EMILY_BENCH_GROUP_SIZE (64).
#
# Greedy decode is deterministic, so before/after (dequant+dot vs fused)
# generate the identical token sequence — the wall-clock RATIO is exact
# even if generation stops before max_new_tokens.

Nx.global_default_backend(Emily.Backend)

env_int = fn name, default ->
  case System.get_env(name) do
    nil -> default
    s -> case Integer.parse(s), do: ({n, _} -> n; _ -> default)
  end
end

repo = System.get_env("EMILY_BENCH_MODEL", "Qwen/Qwen3-0.6B")
prompt = System.get_env("EMILY_BENCH_PROMPT", "The quick brown fox jumps over the lazy dog.")
new_tokens = env_int.("EMILY_BENCH_NEW_TOKENS", 64)
warmup = env_int.("EMILY_BENCH_WARMUP", 1)
runs = env_int.("EMILY_BENCH_RUNS", 3)
group_size = env_int.("EMILY_BENCH_GROUP_SIZE", 64)

IO.puts("Emily / Qwen3 QUANTIZED (affine 4-bit, group_size=#{group_size}) throughput")
IO.puts("  model      : #{repo}")
IO.puts("  new tokens : #{new_tokens}   warmup: #{warmup}   runs: #{runs}")
IO.puts("  lane       : native (Emily.Compiler, native: true, native_fallback: :raise)\n")

{:ok, model_info} = Bumblebee.load_model({:hf, repo})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
{:ok, gen_config} = Bumblebee.load_generation_config({:hf, repo})

gen_config =
  Bumblebee.configure(gen_config,
    max_new_tokens: new_tokens,
    strategy: %{type: :greedy_search}
  )

# EMILY_BENCH_QUANTIZE=0 skips quantization → a dense-native calibration
# lane (anchors what "as fast as it should be" looks like on this host).
quantize? = System.get_env("EMILY_BENCH_QUANTIZE", "1") != "0"

serving_model_info =
  if quantize? do
    IO.puts("Quantizing dense layers -> quantized_dense (%QuantizedWeight{})…")

    {qmodel, qparams} =
      Emily.Quantization.Transform.quantize(model_info.model, model_info.params,
        bits: 4,
        group_size: group_size,
        transpose: true
      )

    %{model_info | model: qmodel, params: qparams}
  else
    IO.puts("Dense (no quantization) calibration lane")
    model_info
  end

serving =
  Bumblebee.Text.generation(serving_model_info, tokenizer, gen_config,
    defn_options: [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  )

for _ <- 1..warmup do
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
secs_list = Enum.map(measurements, fn {secs, _, _} -> secs end)
[{_, _, sample} | _] = measurements
mean = Enum.sum(tps_list) / length(tps_list)
{min_tps, max_tps} = Enum.min_max(tps_list)
median_secs = secs_list |> Enum.sort() |> Enum.at(div(length(secs_list), 2))

IO.puts("\ntokens/sec  : mean=#{Float.round(mean, 2)}  min=#{Float.round(min_tps, 2)}  max=#{Float.round(max_tps, 2)}")
IO.puts("median secs : #{Float.round(median_secs, 4)} (use the ratio of this across before/after runs)")
IO.puts("sample      :\n  #{String.slice(sample, 0, 300)}")
