# Emily

Elixir bindings and Nx backend for Apple's
[MLX](https://github.com/ml-explore/mlx).

## Overview

Emily runs `Nx` computations on Apple Silicon through MLX. Installing
it as the default Nx backend is enough to get Bumblebee models
executing on the Metal GPU with no further integration work —
DistilBERT, Qwen3, ViT, and Whisper all run against pinned reference
outputs in the conformance suite today.

The library is structured as four thin layers, each independently
testable against its own oracle:

```
Emily.Compiler    (Nx.Defn.Compiler) — validates opts, pins the result backend
Emily.Backend     (Nx.Backend)       — op-by-op translation to Native
Emily.Native      (NIF shim)         — one function per MLX op, no policy
MLX C++                              — statically linked into libemily; mlx.metallib alongside
```


## Installation

Add `:emily` to your `mix.exs` deps:

```elixir
def deps do
  [
    {:emily, "~> 0.7"}
  ]
end
```

On first `mix compile` Emily downloads the precompiled NIF for your
OS/arch/variant (`libemily.{so,dylib}` + `mlx.metallib`) from this
repo's GitHub releases into `$EMILY_CACHE` (default
`$(getconf DARWIN_USER_CACHE_DIR)emily/` on macOS,
`${XDG_CACHE_HOME:-~/.cache}/emily/` on Linux) and drops it into
`priv/`. No cmake, Xcode, or C++ toolchain is required on the
consumer side — nothing is compiled locally. See
[Building](#building) for details.

Run `mix emily.doctor` after compiling to verify the host platform,
configured MLX variant, native artifacts, NIF loadability, and a tiny
Emily backend smoke test.

## Features

- **Nx backend.** Every `Nx.*` op dispatches to MLX; ops without a
  native primitive fall back transparently to `Nx.BinaryBackend`
  with a `[:emily, :fallback, *]` telemetry event. See the
  `Fallbacks` section of `Emily.Backend` for the per-op catalogue.
- **Defn compiler.** `Emily.Compiler` runs `defn` / `Nx.Serving` /
  Bumblebee inference on MLX. The default mode dispatches op-by-op
  through `Emily.Backend`; opt in to `native: true` for a single-NIF
  replay of the whole graph (~5× decode throughput on Qwen3-0.6B), and
  `fuse: true` for `mx::compile` kernel fusion on top. See
  [Native compilation](#native-compilation).
- **Fused transformer kernels.** `Emily.Fast` exposes
  `mx::fast::rms_norm`, `layer_norm`, `rope`, and scaled-dot-product
  attention as defn-callable helpers with composed-defn fallbacks for
  other backends.
- **Affine group-wise quantization.** `Emily.QuantizedWeight` +
  `Emily.Quantization` wrap MLX `quantize` / `dequantize` /
  `quantized_matmul` for int2 / int4 / int8 inference. Includes a
  defn-native `dequantize_defn/1` for quantized layers inside Axon
  forward passes.
- **Mixed-precision training.** `Emily.MixedPrecision` provides the
  bf16 recipe (cast params for the forward, keep f32 master, dynamic
  loss scaling with overflow detection).
- **Per-process Metal streams.** `Emily.Stream` lets each BEAM
  process own its own Metal command queue, so multiple processes can
  share a model and run inference concurrently.
- **Zero-copy `to_binary`.** `Nx.to_binary/1` on an Emily tensor
  returns a BEAM resource binary aliasing the MLX buffer — no memcpy.
- **Telemetry.** `[:emily, :eval, *]`, `[:emily, :to_binary, *]`,
  `[:emily, :fallback, *]`, and `[:emily, :memory, :stats]` span
  events. See `Emily.Telemetry`.
- **Compile-time debug flags.** `:debug_bounds_check` and
  `:debug_detect_nan_inf` re-enable runtime assertions on hot paths
  that GPU backends skip by default. Both default off with zero
  runtime cost.

## Prerequisites

- **macOS on Apple Silicon (arm64).** Emily's precompiled NIFs are
  arm64 macOS only; x86_64 Macs aren't supported.
- **Elixir 1.18+ / OTP 27+.** Development is pinned to Elixir 1.19.5
  / OTP 28.3 via `.tool-versions`.

No Xcode, Metal toolchain, cmake, or C++ compiler is required on the
consumer side — Emily downloads a precompiled NIF from GitHub
Releases on first `mix compile`.

## Building

### As a hex consumer

Add `{:emily, "~> 0.7"}` to `mix.exs`, then:

```sh
mix deps.get
mix compile
```

On a cold build Emily downloads the matching precompiled tarball
(`emily-nif-<version>-<variant>-<target>.tar.gz`) from this repo's
GitHub release for the pinned version, verifies its SHA256 against
the checksum pinned in `native_checksums.txt` — a file shipped
*inside* the hex package, so the trust root is covered by the
consumer's `mix.lock` package hash rather than the mutable GitHub
release — and extracts `libemily.{so,dylib}` + `mlx.metallib` into
`priv/`. Extraction goes through `:erl_tar` against a strict entry
allowlist (rejecting symlinks, hardlinks, `..` traversal, absolute
paths, and unexpected entries) rather than shelling out to `tar`.
Subsequent builds reuse the cached tarball under `$EMILY_CACHE`
(default `$(getconf DARWIN_USER_CACHE_DIR)emily/` on macOS,
`${XDG_CACHE_HOME:-~/.cache}/emily/` on Linux), re-verifying it
against the pinned checksum so a republish with a new checksum
invalidates a stale cache automatically.

Override the cache location with `EMILY_CACHE=/some/path mix compile`.

### From source (contributors)

```sh
git clone https://github.com/ausimian/emily.git
cd emily
mix deps.get    # also clones ml-explore/mlx into deps/mlx_src at the pinned tag
mix compile
```

The in-repo checkout keeps `c_src/` on disk, so `mix compile` takes
the source-build path: `scripts/build-mlx.sh` cmake-builds
`libmlx.a` + `mlx.metallib` out of `deps/mlx_src/` into
`$EMILY_CACHE/mlx-<version>-<variant>/`, then `elixir_make` links
the NIF against it. Xcode + the Metal toolchain are required.

Force an MLX rebuild with `mix compile.emily_mlx --force` after
editing `scripts/build-mlx.sh` or bumping `@mlx_version`.

### MLX JIT (optional)

MLX can ship its Metal kernels either AOT-compiled into
`mlx.metallib`, or as source strings that are JIT-compiled on first
use. Emily defaults to the AOT variant. To switch, add to
`config/config.exs`:

```elixir
config :emily, variant: :jit
```

This changes which prebuilt tarball is downloaded; both variants can
coexist under `$EMILY_CACHE`.

Artefact sizes on an M-series Mac (release optimisations):

| Mode                            | `libemily.so` | `mlx.metallib` | `priv/` total |
| ------------------------------- | ------------: | -------------: | ------------: |
| `:aot` (default)                |       ~20 MB  |       ~154 MB  |      ~175 MB  |
| `:jit`                          |       ~22 MB  |       ~3.5 MB  |       ~25 MB  |

With JIT on, kernels are compiled on first invocation, so there's a
small per-kernel warm-up cost at runtime; subsequent calls are cached
in-process. All of Emily's test suite passes under both variants.

The JIT prebuilt is built against the macOS 26.2+ SDK (MLX's NAX
kernel sources transitively include
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, which
only ships in that SDK) and references half-float intrinsics such as
`__fmaxf16` that aren't present in older macOS libSystem. It
therefore also requires **macOS 26.2+ at runtime**; older macOS hosts
should stick with the default `:aot` variant, whose prebuilt is
built against macOS 14 and runs anywhere.

## Usage

Install Emily as the global Nx backend and use Nx normally:

```elixir
Nx.global_default_backend({Emily.Backend, device: :gpu})

Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
|> Nx.dot(Nx.tensor([[5.0], [6.0]]))
|> Nx.to_flat_list()
# => [17.0, 39.0]
```

Use `Emily.Compiler` for `defn` / `Nx.Serving`:

```elixir
Nx.Defn.global_default_options(compiler: Emily.Compiler)
```

Bumblebee inference works with no further configuration once the
backend is installed — see the conformance suites under
`test/emily/conformance/` for worked DistilBERT, Qwen3, ViT, and
Whisper pipelines, and the Livebooks section of the HexDocs nav for
runnable Livebooks.

The low-level tensor API (`Emily.from_binary/3`, `to_binary/1`,
`shape/1`, `dtype/1`, `eval/1`) remains available for diagnostics
and direct MLX round-trips, but most users should go through Nx.

## Native compilation

`Emily.Compiler` has three modes. `native` is **on by default**; opt
into a different mode per-call, globally via
`Nx.Defn.global_default_options/1`, or app-wide with
`config :emily, native: false | true` (a per-call `native:` option
always overrides the app-env default):

```elixir
# Default: native single-NIF replay. Lowers the traced Nx.Defn.Expr to
# a flat IR and replays the whole forward graph in one NIF call per
# invocation — ~5× decode throughput on Qwen3-0.6B, bit-identical to
# the evaluator. Anything the IR can't lower routes through
# Nx.Defn.Evaluator under the default `native_fallback: :eval` (with a
# `[:emily, :compiler, :fallback]` telemetry event).
Nx.Defn.jit(&forward/1, compiler: Emily.Compiler).(input)

# Op-by-op dispatch through Emily.Backend — opt out of native with
# `native: false`. Bit-identical to the Nx evaluator, with no
# whole-graph compile step, so no one-shot compile-time memory spike;
# use it as a numerics reference or on memory-constrained hosts.
Nx.Defn.jit(&forward/1, compiler: Emily.Compiler, native: false).(input)

# Native + mx::compile kernel fusion. Fuses elementwise runs the plain
# replay leaves as separate kernels (RMSNorm / softmax / SiLU gating /
# residual adds) — ~1.1× over the plain native lane on Qwen3-0.6B
# greedy decode (~5.4× the evaluator overall). For a `defn while`,
# the loop body is fused under mx::compile and cached per stream so
# it cache-hits across iterations.
Nx.Defn.jit(&forward/1, compiler: Emily.Compiler, native: true, fuse: true).(input)
```

Trade-off for `:fuse`: `mx::compile` reassociates f32, so logits drift
by a few ULP and the output is **not** bit-identical to the evaluator.
Greedy argmax is empirically robust to that drift (Qwen3-0.6B token
ids matched the evaluator's exactly in our benchmarks), but
**sampling strategies will diverge** from the evaluator even with a
fixed seed.

**Choosing a mode.** `native` is the default and the right one for
model inference: it's bit-identical to the evaluator, safe globally
(un-lowerable ops route through the evaluator), and the single biggest
win — eager → native is the largest jump in every
[benchmark](#performance) tier. Opt out with `native: false` only when
you need the op-by-op evaluator — a numerics reference, or a
memory-constrained host where the one-shot compile peak is too large.
`fuse: true` is **not** a universal
add-on. It pays off only when the fused callable is *reused*, which in
practice means autoregressive decode: the `defn while` body is
`mx::compile`d once and cache-hits every step (the best lane on
Qwen3-0.6B decode, 1.67× EXLA). On a one-shot forward pass — a single
encoder/classifier/vision call — there's nothing to amortize the
`mx::compile` build cost against, so `fuse` is neutral-to-slightly
negative (cf. ViT-base and Whisper-tiny in §5–6 of the benchmark).
Rule of thumb: **`fuse` for decode loops, plain `native` for one-shot
forwards.**

Pass `native_fallback: :raise` to fail rather than silently degrade to
the Evaluator — the conformance suites use this to prove a model
lowers fully native. See the `Emily.Compiler` moduledoc for the full
option list (`:device`, `:hooks`, `:max_concurrency`, etc.) and the
trade-offs around `:fuse` in depth.

For autoregressive decode specifically, `Emily.Generation` is a
model-agnostic loop driver that JIT-compiles a caller-supplied
shape-stable per-token forward and runs the loop from Elixir with
KV-cache threading, stop conditions, and per-token streaming.

## Performance

Emily targets **GPU-friendly model inference on Apple Silicon**. The
[benchmark][bench-report] compares Emily (MLX, Metal GPU) against two
baselines: **EXLA** — which on macOS arm64 ships no GPU client and runs
on the **CPU** — and **EMLX**, the older MLX-backed Nx backend, which
like Emily runs on the **Metal GPU**. EXLA answers the cross-hardware
question (is the GPU faster than XLA on the CPU here?); EMLX answers the
same-hardware one (is Emily's compiler/runtime faster than the older
MLX-backed Nx stack?).

Against EXLA, the practical choice most Elixir-on-Apple-Silicon users
face is GPU-via-Emily vs CPU-via-EXLA, and the two have opposite cost
structures: the GPU has a higher fixed per-op latency floor
(~160–280 µs — a BEAM↔worker hop, a Metal command-buffer commit, and a
GPU sync) but far more throughput at scale; the CPU has a low
(~80–110 µs) floor but caps out sooner. The crossover is **per-op
tensor size**, not model kind:

| Workload (M4 Pro, f32)                   | Best Emily lane vs EXLA-CPU |
| ---------------------------------------- | --------------------------- |
| Large matmul (2048²)                     | **5.0× faster**             |
| ViT-base image classification            | **2.35× faster**            |
| Qwen3-0.6B greedy decode                 | **1.67× faster** (`fuse`)   |
| DistilBERT QA (one encoder forward)      | **1.27× faster**            |
| Whisper-tiny transcription               | **11× slower**              |
| Elementwise / matmul ≤ ~512 per dim      | up to ~2.3× slower          |

**Rule of thumb: reach for Emily when the per-op tensors are large** —
hidden dim ≥ 768 and matmuls ≥ 1024 per dimension, which covers Qwen3,
ViT, and most modern transformers at real sizes. Small models built
from many tiny kernels (Whisper-tiny, hidden dim 384) or workloads
dominated by small tensors can be *slower* than EXLA-CPU, because the
GPU's per-kernel launch latency dominates and its parallelism goes
unused. Notably, every model in the benchmark lowered **fully native
with zero fallbacks**, so these gaps are kernel/dispatch efficiency,
not coverage holes.

**Versus EMLX (GPU-vs-GPU).** The same benchmark runs an `emlx` lane —
the older MLX-backed Nx backend, also on the Metal GPU — so the
comparison isn't only against the CPU. Here Emily's *compiler* is the
differentiator: eager Emily is roughly EMLX-like, but native/fuse pull
far ahead — **2.72× faster on DistilBERT QA**, **5.82× faster on
Qwen3-0.6B decode**, and ~3.2× faster on the Qwen3-4B addendum. (EMLX
did not complete the ViT-base or Whisper-tiny tiers in this harness.)

For decode, use the native compiler rather than the eager backend:
eager Qwen3 decode is 3.2× *slower* than EXLA, while native is 1.5×
faster and fuse 1.67× — a 5.3× throughput swing from eager to fuse
(12.51 → 66.42 tok/s) that is purely the per-op dispatch floor, paid
once per tiny op across thousands of decode steps.

Re-run the benchmark with `elixir bench/emily_vs_exla.exs`; the full
write-up is in [`bench/emily_vs_exla_report.md`][bench-report] and the
raw numbers in [`bench/emily_vs_exla_results.md`][bench-results].

[bench-report]: https://github.com/ausimian/emily/blob/main/bench/emily_vs_exla_report.md
[bench-results]: https://github.com/ausimian/emily/blob/main/bench/emily_vs_exla_results.md

## Livebooks

End-to-end Livebooks under `livebooks/`. Each one declares its own
`Mix.install/2` block and pins `Emily.Backend` as the default Nx
backend, so they're self-contained — open in Livebook and run.

> **Requires Apple Silicon + macOS 14+** (MLX / Metal) — these
> notebooks won't run on Linux or Intel hosts, including the hosted
> Livebooks a "Run in Livebook" badge would open. To get a copy, open
> its page in the Livebooks section of these docs and use the **View
> Source** link (top-right) to view or download the `.livemd` from
> GitHub, then run it in Livebook on a Mac.

- **`distilbert_qa.livemd`** — question answering with
  `distilbert-base-uncased-distilled-squad`.
- **`qwen3_quantized.livemd`** — Qwen3-0.6B int4-quantized via the
  `Emily.Quantization` stack, with concurrent serving over
  `Emily.Stream`.
- **`nomic_embeddings.livemd`** — `nomic-embed-text-v1` sentence
  embeddings on the new Bumblebee 0.7 NomicBERT family, with mean
  pooling, L2 normalisation, and a cosine-similarity demo.
- **`smollm3_chat.livemd`** — chat completion against
  `HuggingFaceTB/SmolLM3-3B` (new in Bumblebee 0.7), including a
  toggle for SmolLM3's hybrid reasoning mode.
- **`modernbert_classification.livemd`** — sequence classification
  with a ModernBERT NLI fine-tune (new in Bumblebee 0.7) — the
  first encoder in the suite with RoPE and alternating
  local/global attention.
- **`mnist_training.livemd`** — Axon training loop with bf16 mixed
  precision via `Emily.MixedPrecision`.
- **`whisper_transcription.livemd`** — Whisper speech-to-text
  against canned audio clips or live microphone input.
- **`fast_kernels.livemd`** — direct use of the fused
  transformer kernels exposed by `Emily.Fast`.

## Concurrency model

MLX dispatches GPU work through Metal command queues. Emily owns one
worker thread per command queue; each worker is a dedicated OS thread
that runs the MLX ops on behalf of BEAM processes. NIFs return
immediately after enqueueing their work on a worker: the worker runs
the op, then posts `{ref, {:ok, result}}` back to the caller via
`enif_send`, and the caller's public wrapper awaits that message with
a plain `receive`. No BEAM scheduler (regular or dirty) blocks on MLX
work — callers see the same synchronous semantics as before, but the
scheduler is free to run other processes while the GPU is busy.

Each worker's queue is **bounded** by `config :emily,
worker_queue_limit: N` (default `8192`); once full, further enqueues
are rejected rather than growing the queue without limit (and pinning
host/GPU memory). A worker that is stopped or dropped replies
`{:error, :stopped}` to everything still queued instead of leaving
those callers blocked. `config :emily, await_timeout: ms` (default
`:infinity`) optionally caps how long a caller waits for a native
result.

Because the MLX stream is pinned to its worker thread, MLX's
per-thread `CommandEncoder` state stays consistent regardless of how
the BEAM migrates Elixir processes between schedulers.

By default, every op uses the **default worker** owned by the
`Emily.MlxStream.Default` GenServer under the application supervisor.
That single queue serialises all GPU work across the VM — correct
and simple, but a bottleneck under concurrent inference.

### Stream-per-worker, shared weights

The recommended pattern for concurrent inference: load the model
**once**, create a **pool of streams** at boot, and route each
request to a worker that owns one of those streams. Weights live in
one MLX buffer that every stream reads; the only per-stream cost is
the Metal command buffer.

```elixir
# 1. Load weights once, at application start.
{:ok, model}     = Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})
{:ok, config}    = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-0.6B"})

serving =
  Bumblebee.Text.generation(model, tokenizer, config,
    defn_options: [compiler: Emily.Compiler]
  )

# 2. Start N workers; each owns one Emily.Stream for its lifetime.
defmodule MyApp.StreamWorker do
  use GenServer

  def start_link({id, serving}),
    do: GenServer.start_link(__MODULE__, serving, name: via(id))

  def run(id, input), do: GenServer.call(via(id), {:run, input}, :infinity)

  @impl true
  def init(serving) do
    {:ok, %{serving: serving, stream: Emily.Stream.new(:gpu)}}
  end

  @impl true
  def handle_call({:run, input}, _from, %{stream: s, serving: sv} = state) do
    result = Emily.Stream.with_stream(s, fn -> Nx.Serving.run(sv, input) end)
    {:reply, result, state}
  end

  defp via(id), do: {:via, Registry, {MyApp.StreamRegistry, id}}
end

# 3. Dispatch each request to any free worker (round-robin, poolboy,
#    a Registry lookup, etc.). Calls to different workers run
#    concurrently on distinct Metal command queues.
MyApp.StreamWorker.run(pick_worker(), "The quick brown fox…")
```

Create streams once at worker init, not per-request —
`Emily.Stream.new/1` spawns an OS thread.

**Stream lifecycle.** `Emily.Stream.close/1` stops a stream's worker
deterministically: it cancels any still-queued ops (their awaiting
callers get a `RuntimeError`), lets the in-flight op finish, and then
joins the OS thread. Closing is optional — if you never call it,
cleanup piggybacks on BEAM GC of the NIF resource held in the struct.
In the pattern above, the stream lives as long as its owning worker
process: when the worker terminates (crash, supervisor shutdown, or
`GenServer.stop/1`), the process heap is reclaimed, the resource's
refcount drops to zero, and the worker is torn down. Either way the
OS thread is joined **off the BEAM schedulers** by a dedicated reaper,
so collecting (or closing) a busy stream never stalls a scheduler. A
supervised restart drops the old stream and allocates a fresh one in
the child's `init/1`. To drop a stream deliberately, call
`Emily.Stream.close/1` or terminate the process that owns it.

### Pooled servings — K weight copies, one default queue

For small models where duplicating weights is cheap, start K
`Nx.Serving` instances behind poolboy / Registry / etc. Each instance
holds its own copy of the weights. No `Emily.Stream` is involved, so
**every instance dispatches onto the same default Metal command
queue** — requests run sequentially at the GPU even though multiple
servings exist at the BEAM level. The pool buys parallelism for
CPU-side serving work (pre/post-processing, batching) but not for
GPU-side compute. Memory scales linearly with K.

Combine the two if you need both: K servings for CPU parallelism,
each with its own `Emily.Stream` for GPU parallelism.

### Using Emily with `Nx.Serving`

`Nx.Serving` itself is stream-agnostic — it calls into `Emily.Compiler`
which dispatches to whatever MLX stream is installed in the calling
process. That gives three viable configurations:

| Configuration                                | Weights in GPU memory | GPU queues   | When to use                                                   |
| -------------------------------------------- | --------------------- | ------------ | ------------------------------------------------------------- |
| Single serving, default stream               | 1×                    | 1 (shared)   | Default. Simplest; fine for single-user / batched workloads.  |
| Single serving + pool of `Emily.Stream`s     | 1×                    | N (per ws)   | Concurrent inference on a shared model. Large models.         |
| K servings (pooled), default stream          | K×                    | 1 (shared)   | Small models where CPU serving work dominates GPU compute.    |

In every case `Nx.Serving.run/2` / `Nx.Serving.batched_run/2` is the
caller-facing API; the only difference is whether the calling
process wraps the call in `Emily.Stream.with_stream/2` and whether
you run one serving or many.

See `Emily.Stream` for the API and the `qwen3_quantized` livebook
under Livebooks for a worked multi-stream example.

## Observability

Emily emits `:telemetry` events at the evaluation boundary
(`[:emily, :eval, *]`, `[:emily, :to_binary, *]`) and at every
`Nx.BinaryBackend` fallback (`[:emily, :fallback, *]`). Attach a
handler to graph hotspots or detect silent performance cliffs —
see `Emily.Telemetry` for the full event catalogue.

When a backend callback has no native MLX path, Emily transparently
falls back to `Nx.BinaryBackend`. The fallback is ~100× slower;
configure per-fallback behaviour with `:fallback`:

```elixir
# config/dev.exs — one-shot Logger.warning per {op, shapes} pair
config :emily, fallback: :warn

# config/test.exs (CI) — fail loud if a hot path goes via BinaryBackend
config :emily, fallback: :raise
```

The default is `:silent`, so library consumers and CI logs stay quiet
unless they opt in. The `[:emily, :fallback, *]` telemetry events
fire regardless in `:silent`/`:warn` mode; `:raise` raises on entry
and skips the span.

The legacy `config :emily, :warn_on_fallback, true` boolean is still
honoured when `:fallback` is unset (`true` → `:warn`). Prefer
`:fallback` in new code.

### Memory

MLX buffers live outside the BEAM heap, so long-running serving and
training processes should observe MLX allocator state directly.
`Emily.Memory.stats/0` returns active, peak, and cached bytes and emits
the same `[:emily, :memory, :stats]` telemetry event:

```elixir
stats = Emily.Memory.stats()
# %{active: ..., peak: ..., cache: ...}
```

Use `Emily.Memory.reset_peak/0` before a benchmark or soak window, and
`Emily.Memory.clear_cache/0` when you want MLX to release reusable
cached buffers. `clear_cache/0` does not free live tensors or binaries
returned by `Nx.to_binary/1`; those buffers are released only after the
owning BEAM references are garbage collected.

## Debug assertions

Two compile-time flags re-enable runtime checks that MLX (and every
other GPU backend) skips by default. Both are off by default with
zero runtime cost when off — the guarded branches are dead-code
eliminated by the Elixir compiler.

```elixir
# config/dev.exs
config :emily,
  debug_bounds_check: true,
  debug_detect_nan_inf: true
```

- `:debug_bounds_check` — raises on out-of-range / negative indices
  in `gather` / `take` / `take_along_axis` / `indexed_add` /
  `indexed_put`. Catches the silent-`NaN`-from-OOB-gather class of
  bug (e.g. a vocab-30522 tokenizer paired with a tiny-random model
  whose embedding table is smaller).
- `:debug_detect_nan_inf` — scans results of `matmul`, the fused
  `layer_norm` / `rms_norm`, and both fused SDPA variants. Surfaces
  numerics failures at the producing op rather than downstream.

Each check is a per-op MLX reduction plus a scalar readback — a
worker sync that breaks lazy-graph fusion. Leave off in release
builds. See the `Emily` moduledoc for the full opt-in snippet.

## Testing

```bash
mix test                           # fast suite (unit + property)
mix test --only conformance        # + Bumblebee tiny-random suites
mix test --only qwen3_full         # full Qwen3-0.6B checkpoint (~1.5 GB)
mix test --only qwen3_quant_full   # quantized Qwen3-0.6B end-to-end
mix test --only vit_full           # full ViT-base (~330 MB)
mix test --only whisper_full       # full whisper-tiny (~150 MB)
mix test --only training_full      # MNIST convergence canary
mix test --only soak               # memory + concurrency soak harnesses
```

Each layer has its own oracle: hand-computed expected values at the
Native layer, `Nx.BinaryBackend` on the same inputs at the Backend
layer, `Emily.Backend` in non-defn mode at the Compiler layer, and
HuggingFace Transformers reference slices end-to-end. A bug can only
be introduced in the layer where its test fails.

## Documentation

  * [HexDocs](https://hexdocs.pm/emily) — per-module API docs and
    runnable livebooks.
  * [`ARCHITECTURE.md`](ARCHITECTURE.md) — current shape of the
    library: layer boundaries, design decisions, concurrency and
    memory model, observability surface.
  * [`ROADMAP.md`](ROADMAP.md) — active and future work, including
    items deferred to post-1.0.
  * [`PLAN.md`](https://github.com/ausimian/emily/blob/main/PLAN.md) — historical milestone log. The "why" behind
    a current behaviour usually lives here, named by milestone.

## Acknowledgements

Emily stands on the shoulders of two projects without which it would not exist:

- **[MLX](https://github.com/ml-explore/mlx)** — Apple's array framework for
  Apple Silicon, developed by the MLX team at Apple. Emily is a thin Elixir
  layer over MLX's C++ API; every tensor op ultimately runs through MLX.
- **[EMLX](https://github.com/elixir-nx/emlx)** — the original Elixir-Nx
  MLX backend by [Paulo Valente](https://github.com/polvalente) and
  contributors. EMLX proved the integration shape between Nx and MLX, and
  Emily used it as a design reference and early ground-truth source.

Thanks also to [Cocoa Xu](https://github.com/cocoa-xu), whose MLX build
scripts and prebuilt-binary tooling underpin much of the macOS NIF supply
chain in the Elixir ML ecosystem.

## License

MIT
