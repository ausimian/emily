# Emily (MLX / Metal GPU) vs EXLA (XLA / CPU) performance comparison.
#
# Produces a like-for-like benchmark of the local, unpublished Emily backend
# against the EXLA backend across five tiers:
#
#   1. Op microbenchmarks — add / multiply / exp / sum / matmul / softmax over
#      a range of square f32 tensors. Measures per-op kernel + dispatch cost.
#   2. DistilBERT question-answering — a full encoder forward per call.
#   3. Qwen3-0.6B greedy decode — autoregressive token throughput.
#   4. ViT-base image classification — a vision encoder forward per call.
#   5. Whisper-tiny transcription — mel featurizer + encoder + bounded decode.
#
# (Quantization is intentionally excluded: Emily's int4/int8 path is MLX
# `quantized_matmul`, which EXLA has no equivalent for, so there is no
# like-for-like quantized lane to compare against.)
#
# Lanes compared in every tier:
#
#   * exla         — `compiler: EXLA`. On Apple Silicon EXLA has no GPU target,
#                    so this is the XLA **CPU** backend. This is the only EXLA
#                    lane that exists on this platform.
#   * emily-eager  — `Emily.Backend` under `Nx.Defn.Evaluator`: op-by-op
#                    dispatch onto the Metal GPU, one BEAM<->worker hop per op.
#   * emily-native — `Emily.Compiler, native: true`: the whole traced graph
#                    replays as one NIF call (the single-NIF lane).
#   * emily-fuse   — native plus `fuse: true`: `mx::compile` kernel fusion on
#                    top. NOT bit-identical (f32 reassociation), reported as an
#                    extra lane.
#
# IMPORTANT FRAMING: this is a GPU (Emily/MLX) vs CPU (EXLA/XLA-host)
# comparison, not GPU-vs-GPU. EXLA on macOS arm64 ships only the host/CPU
# PJRT client. The report interprets results in that light.
#
# Usage:
#
#     elixir bench/emily_vs_exla.exs                 # full suite
#     EMILY_BENCH_SMOKE=1 elixir bench/emily_vs_exla.exs   # quick validation
#
# Env knobs:
#
#     EMILY_BENCH_SMOKE         "1" -> tiny op-only run (validate harness).
#     EMILY_BENCH_OP_ITERS      measured iters per op/size   (default 30).
#     EMILY_BENCH_OP_WARMUP     warmup iters per op/size      (default 5).
#     EMILY_BENCH_TOKENS        new tokens for Qwen3 decode   (default 48).
#     EMILY_BENCH_WHISPER_TOKENS max new tokens for Whisper   (default 25).
#     EMILY_BENCH_RUNS          measured runs for model tiers (default 3).
#     EMILY_BENCH_SKIP          comma list of tiers to skip:
#                               ops,distilbert,qwen3,vit,whisper.
#     EMILY_BENCH_OUT           results markdown path
#                               (default bench/emily_vs_exla_results.md).
#
# First run downloads the XLA binary for EXLA (~hundreds of MB) and compiles
# the Emily NIF from the local checkout; both are cached afterwards. Models are
# read from the existing Bumblebee cache.

Mix.install([
  {:emily, path: Path.expand("..", __DIR__)},
  {:exla, "~> 0.12"},
  {:bumblebee, "~> 0.7"},
  {:axon, "~> 0.8"},
  {:tokenizers, "~> 0.5"},
  {:nx, "~> 0.12"}
])

# ---------------------------------------------------------------------------
# defn kernels for the op microbenchmark. Each returns a scalar (sum of the
# op output) so realization is forced identically across backends and the
# host readback is a single number on every lane.
# ---------------------------------------------------------------------------
defmodule Emily.Bench.Kernels do
  import Nx.Defn

  defn add_(a, b), do: Nx.sum(Nx.add(a, b))
  defn mul_(a, b), do: Nx.sum(Nx.multiply(a, b))
  defn exp_(a), do: Nx.sum(Nx.exp(a))
  defn sum_(a), do: Nx.sum(a)
  defn matmul_(a, b), do: Nx.sum(Nx.dot(a, b))

  defn softmax_(a) do
    m = Nx.reduce_max(a, axes: [-1], keep_axes: true)
    e = Nx.exp(a - m)
    Nx.sum(e / Nx.sum(e, axes: [-1], keep_axes: true))
  end
end

defmodule Emily.Bench.ExlaVsEmily do
  alias Emily.Bench.Kernels

  # Lane = {id, label, default_backend, defn_options}. The default backend is
  # where inputs/params live so no per-call host<->device transfer is timed.
  @lanes [
    {:exla, "exla (CPU)", EXLA.Backend, [compiler: EXLA]},
    {:emily_eager, "emily-eager (GPU)", Emily.Backend, [compiler: Nx.Defn.Evaluator]},
    {:emily_native, "emily-native (GPU)", Emily.Backend,
     [compiler: Emily.Compiler, native: true]},
    {:emily_fuse, "emily-fuse (GPU)", Emily.Backend,
     [compiler: Emily.Compiler, native: true, fuse: true]}
  ]

  def run do
    smoke? = System.get_env("EMILY_BENCH_SMOKE") == "1"
    skip = (System.get_env("EMILY_BENCH_SKIP") || "") |> String.split(",", trim: true)
    out = System.get_env("EMILY_BENCH_OUT") || Path.expand("emily_vs_exla_results.md", __DIR__)

    op_iters = env_int("EMILY_BENCH_OP_ITERS", if(smoke?, do: 5, else: 30))
    op_warmup = env_int("EMILY_BENCH_OP_WARMUP", if(smoke?, do: 2, else: 5))
    tokens = env_int("EMILY_BENCH_TOKENS", if(smoke?, do: 8, else: 48))
    whisper_tokens = env_int("EMILY_BENCH_WHISPER_TOKENS", if(smoke?, do: 4, else: 25))
    runs = env_int("EMILY_BENCH_RUNS", if(smoke?, do: 1, else: 3))

    banner("Emily (MLX/GPU) vs EXLA (XLA/CPU)")
    IO.puts("  smoke?      : #{smoke?}")
    IO.puts("  op iters    : #{op_iters} (warmup #{op_warmup})")
    IO.puts("  model runs  : #{runs}, qwen3 tokens: #{tokens}, whisper tokens: #{whisper_tokens}")
    IO.puts("  exla client : #{exla_platform()}")
    IO.puts("  skip        : #{inspect(skip)}\n")

    Process.put(:bench_fallback_ctr, setup_fallback_counter())

    results = %{}

    results =
      if "ops" in skip,
        do: results,
        else: Map.put(results, :ops, bench_ops(smoke?, op_iters, op_warmup))

    results =
      if smoke? or "distilbert" in skip,
        do: results,
        else: Map.put(results, :distilbert, bench_distilbert(runs))

    results =
      if smoke? or "qwen3" in skip,
        do: results,
        else: Map.put(results, :qwen3, bench_qwen3(tokens, runs))

    results =
      if smoke? or "vit" in skip,
        do: results,
        else: Map.put(results, :vit, bench_vit(runs))

    results =
      if smoke? or "whisper" in skip,
        do: results,
        else: Map.put(results, :whisper, bench_whisper(whisper_tokens, runs))

    write_report(out, results, %{
      smoke?: smoke?,
      op_iters: op_iters,
      tokens: tokens,
      runs: runs
    })

    IO.puts("\nWrote results to #{out}")
  end

  # ---------------------------------------------------------------------
  # Tier 1: op microbenchmarks
  # ---------------------------------------------------------------------

  # {name, arity, fun, sizes}. Inputs are square NxN f32 matrices.
  defp op_specs(smoke?) do
    sizes = if smoke?, do: [256], else: [256, 1024, 4096]
    mm_sizes = if smoke?, do: [256], else: [128, 512, 1024, 2048]

    [
      {"add", 2, &Kernels.add_/2, sizes},
      {"mul", 2, &Kernels.mul_/2, sizes},
      {"exp", 1, &Kernels.exp_/1, sizes},
      {"sum", 1, &Kernels.sum_/1, sizes},
      {"softmax", 1, &Kernels.softmax_/1, sizes},
      {"matmul", 2, &Kernels.matmul_/2, mm_sizes}
    ]
  end

  defp bench_ops(smoke?, iters, warmup) do
    banner("Tier 1: op microbenchmarks (mean us/call, op + sum, scalar readback)")

    for {name, arity, fun, sizes} <- op_specs(smoke?), size <- sizes do
      label = "#{name} #{size}x#{size}"
      IO.puts("\n=== #{label} ===")

      lane_us =
        for {id, lane_label, backend, opts} <- @lanes, into: %{} do
          us = measure_op(fun, arity, size, backend, opts, iters, warmup)
          IO.puts("  #{String.pad_trailing(lane_label, 20)} #{fmt_us(us)}")
          {id, us}
        end

      %{op: name, size: size, lanes: lane_us}
    end
  end

  defp measure_op(fun, arity, size, backend, opts, iters, warmup) do
    inputs = op_inputs(arity, size, backend)
    jitted = Nx.Defn.jit(fun, opts)

    call = fn -> apply(jitted, inputs) |> Nx.to_number() end

    safe(fn ->
      for _ <- 1..warmup, do: call.()
      times = for _ <- 1..iters, do: elem(:timer.tc(call), 0)
      Enum.sum(times) / length(times)
    end)
  rescue
    e -> {:error, Exception.message(e)}
  end

  # Deterministic, backend-resident inputs. iota/sin keeps values in a sane
  # range for exp/softmax without RNG (Math.random is unavailable anyway).
  defp op_inputs(arity, size, backend) do
    a = mat(size, 1, backend)
    if arity == 2, do: [a, mat(size, 2, backend)], else: [a]
  end

  defp mat(n, seed, backend) do
    Nx.iota({n, n}, type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.add(seed * 1.0)
    |> Nx.multiply(1.0 / (n * n))
    |> Nx.sin()
    |> Nx.backend_transfer(backend)
  end

  # ---------------------------------------------------------------------
  # Tier 2: DistilBERT question answering (full encoder forward per call)
  # ---------------------------------------------------------------------

  defp bench_distilbert(runs) do
    banner("Tier 2: DistilBERT QA (distilbert-base-uncased-distilled-squad)")
    repo = "distilbert-base-uncased-distilled-squad"

    input = %{
      question: "What is Emily?",
      context:
        "Emily is an Elixir Nx backend that runs tensor computations on " <>
          "Apple Silicon through MLX, executing on the Metal GPU."
    }

    for {id, label, backend, opts} <- @lanes, into: %{} do
      us =
        safe(fn ->
          Nx.global_default_backend(backend)
          {:ok, model} = Bumblebee.load_model({:hf, repo})
          {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
          serving = Bumblebee.Text.question_answering(model, tokenizer, defn_options: opts)
          measure_serving(label, serving, input, runs)
        end)

      reclaim()
      {id, us}
    end
  end

  # ---------------------------------------------------------------------
  # Tier 3: Qwen3-0.6B greedy decode (tokens/sec)
  # ---------------------------------------------------------------------

  defp bench_qwen3(tokens, runs) do
    banner("Tier 3: Qwen3-0.6B greedy decode (#{tokens} new tokens)")
    repo = "Qwen/Qwen3-0.6B"
    prompt = "The quick brown fox jumps over the lazy dog."

    for {id, label, backend, opts} <- @lanes, into: %{} do
      result =
        safe(fn ->
          Nx.global_default_backend(backend)
          {:ok, model} = Bumblebee.load_model({:hf, repo})
          {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
          {:ok, gen_config} = Bumblebee.load_generation_config({:hf, repo})

          gen_config =
            Bumblebee.configure(gen_config,
              max_new_tokens: tokens,
              strategy: %{type: :greedy_search}
            )

          serving =
            Bumblebee.Text.generation(model, tokenizer, gen_config, defn_options: opts)

          # tokens/sec rather than us/call for the decode tier.
          us = measure_serving(label, serving, prompt, runs)
          if is_number(us), do: tokens / (us / 1_000_000), else: us
        end)

      reclaim()
      {id, result}
    end
  end

  # ---------------------------------------------------------------------
  # Tier 4: ViT-base image classification (one forward per call)
  # ---------------------------------------------------------------------

  defp bench_vit(runs) do
    banner("Tier 4: ViT-base image classification (google/vit-base-patch16-224)")
    repo = "google/vit-base-patch16-224"

    # Deterministic synthetic 224x224 RGB image. u8 iota wraps 0..255, so
    # the featurizer (resize is a no-op at 224, then normalize) gets a
    # varied pattern. Label quality is irrelevant — only forward cost is.
    image = Nx.iota({224, 224, 3}, type: :u8, backend: Nx.BinaryBackend)

    for {id, label, backend, opts} <- @lanes, into: %{} do
      us =
        safe(fn ->
          Nx.global_default_backend(backend)
          {:ok, model} = Bumblebee.load_model({:hf, repo})
          {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo})
          serving = Bumblebee.Vision.image_classification(model, featurizer, defn_options: opts)
          measure_serving(label, serving, image, runs)
        end)

      reclaim()
      {id, us}
    end
  end

  # ---------------------------------------------------------------------
  # Tier 5: Whisper-tiny speech-to-text (featurizer + encoder + decode)
  # ---------------------------------------------------------------------

  defp bench_whisper(tokens, runs) do
    banner("Tier 5: Whisper-tiny transcription (#{tokens} max new tokens)")
    repo = "openai/whisper-tiny"

    # ~1 s of deterministic synthetic audio. The featurizer pads it to
    # Whisper's 30 s mel window, so the encoder runs its full 1500-position
    # path regardless of the (short) audio. Latency here is encoder +
    # featurizer dominated; Tier 3 (Qwen3) is the decode-dominated case.
    audio =
      Nx.sin(Nx.iota({16_000}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02))

    for {id, label, backend, opts} <- @lanes, into: %{} do
      us =
        safe(fn ->
          Nx.global_default_backend(backend)
          {:ok, model} = Bumblebee.load_model({:hf, repo})
          {:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo})
          {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
          {:ok, gen_config} = Bumblebee.load_generation_config({:hf, repo})
          gen_config = Bumblebee.configure(gen_config, max_new_tokens: tokens)

          serving =
            Bumblebee.Audio.speech_to_text_whisper(
              model,
              featurizer,
              tokenizer,
              gen_config,
              defn_options: opts
            )

          measure_serving(label, serving, audio, runs)
        end)

      reclaim()
      {id, us}
    end
  end

  # ---------------------------------------------------------------------
  # Shared serving timing
  # ---------------------------------------------------------------------

  defp measure_serving(label, serving, input, runs) do
    IO.puts("  [#{label}] warmup (compile)…")
    _ = Nx.Serving.run(serving, input)

    {bin0, cmp0} = fallback_counts()

    times =
      for n <- 1..runs do
        {us, _} = :timer.tc(fn -> Nx.Serving.run(serving, input) end)
        IO.puts("    [#{label}] run #{n}: #{fmt_us(us)}")
        us
      end

    {bin1, cmp1} = fallback_counts()
    bin = div(bin1 - bin0, runs)
    cmp = div(cmp1 - cmp0, runs)

    # BinaryBackend op fallbacks (binary) and native->evaluator fallbacks
    # (compiler), per call. Nonzero `binary` means ops ran ~100x slower on
    # the CPU BinaryBackend; nonzero `compiler` means the native lowering bailed.
    IO.puts("    [#{label}] fallbacks/call: binary=#{bin} compiler=#{cmp}")

    Enum.sum(times) / length(times)
  end

  # Two-slot counter: slot 1 = `[:emily, :fallback, :stop]` (an op fell back
  # to Nx.BinaryBackend); slot 2 = `[:emily, :compiler, :fallback]` (the native
  # compiler bailed a construct to the evaluator).
  defp setup_fallback_counter do
    ctr = :counters.new(2, [:write_concurrency])

    :telemetry.detach("bench-fallback-binary")
    :telemetry.detach("bench-fallback-compiler")

    :telemetry.attach(
      "bench-fallback-binary",
      [:emily, :fallback, :stop],
      fn _e, _m, _meta, _cfg -> :counters.add(ctr, 1, 1) end,
      nil
    )

    :telemetry.attach(
      "bench-fallback-compiler",
      [:emily, :compiler, :fallback],
      fn _e, _m, _meta, _cfg -> :counters.add(ctr, 2, 1) end,
      nil
    )

    ctr
  end

  defp fallback_counts do
    case Process.get(:bench_fallback_ctr) do
      nil -> {0, 0}
      ctr -> {:counters.get(ctr, 1), :counters.get(ctr, 2)}
    end
  end

  # ---------------------------------------------------------------------
  # Report writer
  # ---------------------------------------------------------------------

  defp write_report(path, results, meta) do
    lines =
      [
        "# Emily vs EXLA — raw benchmark results",
        "",
        "_Auto-generated by `bench/emily_vs_exla.exs`. Lower is better for us/call;",
        "higher is better for tokens/sec. Ratios are relative to the EXLA lane._",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| ----- | ----- |",
        "| date | #{date_string()} |",
        "| host | #{host_string()} |",
        "| elixir / otp | #{System.version()} / #{:erlang.system_info(:otp_release)} |",
        "| emily | #{vsn(:emily)} |",
        "| exla | #{vsn(:exla)} (#{exla_platform()}) |",
        "| nx | #{vsn(:nx)} |",
        "| smoke run? | #{meta.smoke?} |",
        ""
      ] ++
        ops_section(results[:ops]) ++
        serving_section("Tier 2 — DistilBERT QA (mean us/call)", results[:distilbert], :us) ++
        serving_section("Tier 3 — Qwen3-0.6B decode (tokens/sec)", results[:qwen3], :tps) ++
        serving_section("Tier 4 — ViT-base image classification (mean us/call)", results[:vit], :us) ++
        serving_section("Tier 5 — Whisper-tiny transcription (mean us/call)", results[:whisper], :us)

    File.write!(path, Enum.join(lines, "\n") <> "\n")
  end

  defp ops_section(nil), do: []

  defp ops_section(ops) do
    header = [
      "## Tier 1 — op microbenchmarks (mean us/call)",
      "",
      "| op | size | exla (CPU) | emily-eager | emily-native | emily-fuse | best-emily/exla |",
      "| -- | ---- | ---------- | ----------- | ------------ | ---------- | --------------- |"
    ]

    rows =
      for %{op: op, size: size, lanes: lanes} <- ops do
        exla = lanes[:exla]

        best_emily =
          [lanes[:emily_eager], lanes[:emily_native], lanes[:emily_fuse]]
          |> Enum.filter(&is_number/1)
          |> case do
            [] -> nil
            xs -> Enum.min(xs)
          end

        ratio =
          if is_number(exla) and is_number(best_emily),
            do: fmt_ratio(best_emily / exla),
            else: "—"

        "| #{op} | #{size} | #{cell(exla)} | #{cell(lanes[:emily_eager])} | " <>
          "#{cell(lanes[:emily_native])} | #{cell(lanes[:emily_fuse])} | #{ratio} |"
      end

    header ++ rows ++ [""]
  end

  defp serving_section(_title, nil, _kind), do: []

  defp serving_section(title, lanes, kind) do
    exla = lanes[:exla]

    hdr_metric = if kind == :us, do: "us/call", else: "tok/s"
    ratio_hdr = "best Emily lane vs EXLA (>1 = Emily faster)"

    emily_vals =
      [lanes[:emily_eager], lanes[:emily_native], lanes[:emily_fuse]]
      |> Enum.filter(&is_number/1)

    best =
      case {kind, emily_vals} do
        {_, []} -> nil
        {:us, xs} -> Enum.min(xs)
        {:tps, xs} -> Enum.max(xs)
      end

    # Speedup with a consistent direction regardless of metric: for us/call
    # (lower better) that is exla/best; for tok/s (higher better) best/exla.
    ratio =
      cond do
        not is_number(exla) or is_nil(best) -> "—"
        kind == :us -> fmt_ratio(exla / best)
        kind == :tps -> fmt_ratio(best / exla)
      end

    [
      "## #{title}",
      "",
      "| lane | #{hdr_metric} |",
      "| ---- | ------------- |",
      "| exla (CPU) | #{cell_kind(lanes[:exla], kind)} |",
      "| emily-eager (GPU) | #{cell_kind(lanes[:emily_eager], kind)} |",
      "| emily-native (GPU) | #{cell_kind(lanes[:emily_native], kind)} |",
      "| emily-fuse (GPU) | #{cell_kind(lanes[:emily_fuse], kind)} |",
      "",
      "_#{ratio_hdr}: #{ratio}_",
      ""
    ]
  end

  # ---------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------

  defp safe(fun) do
    fun.()
  rescue
    e -> {:error, Exception.message(e)}
  catch
    kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
  end

  # Free GPU/host buffers between model lanes so peak memory stays bounded.
  defp reclaim do
    :erlang.garbage_collect()
    if function_exported?(Emily.Memory, :clear_cache, 0), do: Emily.Memory.clear_cache()
    :ok
  end

  defp cell({:error, _}), do: "ERR"
  defp cell(n) when is_number(n), do: Float.round(n / 1.0, 1) |> Float.to_string()
  defp cell(_), do: "—"

  defp cell_kind(v, :us), do: cell(v)
  defp cell_kind({:error, _}, :tps), do: "ERR"
  defp cell_kind(n, :tps) when is_number(n), do: Float.round(n / 1.0, 2) |> Float.to_string()
  defp cell_kind(_, :tps), do: "—"

  defp fmt_us({:error, msg}), do: "ERROR: #{msg}"
  defp fmt_us(us) when is_number(us), do: "#{Float.round(us / 1.0, 1)} us/call"
  defp fmt_us(_), do: "—"

  defp fmt_ratio(r), do: "#{Float.round(r / 1.0, 2)}x"

  defp banner(text) do
    IO.puts("\n" <> String.duplicate("=", 72))
    IO.puts(text)
    IO.puts(String.duplicate("=", 72))
  end

  defp env_int(name, default) do
    case System.get_env(name) do
      nil ->
        default

      s ->
        case Integer.parse(s) do
          {n, _} -> n
          _ -> default
        end
    end
  end

  defp exla_platform do
    safe(fn ->
      client = EXLA.Client.fetch!(EXLA.Client.default_name())
      "#{client.platform} (#{client.device_count} device)"
    end)
    |> case do
      {:error, _} -> "host/CPU"
      s -> s
    end
  end

  defp vsn(app) do
    case Application.spec(app, :vsn) do
      nil -> "?"
      v -> to_string(v)
    end
  end

  defp date_string do
    {{y, m, d}, {hh, mm, _}} = :calendar.local_time()
    :io_lib.format("~4..0w-~2..0w-~2..0w ~2..0w:~2..0w", [y, m, d, hh, mm])
    |> to_string()
  end

  defp host_string do
    {out, 0} = System.cmd("sysctl", ["-n", "machdep.cpu.brand_string"])
    mem = div(:erlang.memory(:total), 1_000_000)
    "#{String.trim(out)} (BEAM total #{mem} MB at write)"
  end
end

Emily.Bench.ExlaVsEmily.run()
