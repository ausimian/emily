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
MLX C++           (prebuilt)         — libmlx.a + mlx.metallib fetched from GitHub
```


## Installation

Add `:emily` to your `mix.exs` deps:

```elixir
def deps do
  [
    {:emily, "~> 0.1.0"}
  ]
end
```

On first `mix compile` Emily downloads a prebuilt `libmlx.a` +
`mlx.metallib` tarball from this repo's GitHub releases into
`$EMILY_CACHE` (default `~/Library/Caches/emily`) and links it into
the NIF. No cmake or Xcode toolchain is required on the consumer
side. See [Building](#building) for details.

## Features

- **Nx backend.** Every `Nx.*` op dispatches to MLX; ops without a
  native primitive fall back transparently to `Nx.BinaryBackend`
  with a `[:emily, :fallback, *]` telemetry event. See the
  `Fallbacks` section of `Emily.Backend` for the per-op catalogue.
- **Defn compiler.** `Emily.Compiler` runs `defn` / `Nx.Serving` /
  Bumblebee inference on MLX. Backs the results with lazy MLX graphs.
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

- **macOS on Apple Silicon (arm64).** Emily's MLX prebuilts are arm64
  macOS only; x86_64 Macs aren't supported.
- **Elixir 1.18+ / OTP 27+.** Development is pinned to Elixir 1.19.5
  / OTP 28.3 via `.tool-versions`.

No Xcode, Metal toolchain, or cmake is required on the consumer side
— Emily downloads a prebuilt `libmlx.a` + `mlx.metallib` from GitHub.

## Building

```sh
git clone https://github.com/ausimian/emily.git
cd emily
mix deps.get
mix compile
```

On a cold build Emily downloads the pinned MLX prebuilt
(`mlx-<version>-macos-arm64-<variant>.tar.gz`, ~40 MB AOT / ~5 MB JIT)
from this repo's releases into `$EMILY_CACHE` (default
`~/Library/Caches/emily/mlx-<version>-<variant>/`), verifies its
SHA256 against `@mlx_checksums` in `mix.exs`, extracts it, and links
it into the NIF. Subsequent builds reuse the cached install.

Override the cache location with `EMILY_CACHE=/some/path mix
compile`, or re-fetch with `mix compile.emily_mlx --force`.

The MLX version is pinned in `mix.exs` as `@mlx_version`. Bumps
happen in this repo — new prebuilts are produced by the manual
`release-mlx.yml` workflow, which uploads tarballs + SHA256s to a
`mlx-<version>` release on this repo; the `@mlx_version` /
`@mlx_checksums` pair in `mix.exs` is then updated in the same commit.

### MLX JIT (optional)

MLX can ship its Metal kernels either AOT-compiled into
`mlx.metallib`, or as source strings that are JIT-compiled on first
use. Emily defaults to the AOT variant. To switch, add to
`config/config.exs`:

```elixir
config :emily, mlx_variant: :jit
```

This changes which prebuilt tarball is downloaded; both variants can
coexist under `$EMILY_CACHE`.

Artefact sizes on an M-series Mac (release optimisations):

| Mode                            | `libemily.so` | `mlx.metallib` | `priv/` total |
| ------------------------------- | ------------: | -------------: | ------------: |
| `:no_jit` (default)             |       ~20 MB  |       ~154 MB  |      ~175 MB  |
| `:jit`                          |       ~22 MB  |       ~3.5 MB  |       ~25 MB  |

With JIT on, kernels are compiled on first invocation, so there's a
small per-kernel warm-up cost at runtime; subsequent calls are cached
in-process. All of Emily's test suite passes under both variants.

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
Whisper pipelines, and the Notebooks section of the HexDocs nav for
runnable Livebooks.

The low-level tensor API (`Emily.from_binary/3`, `to_binary/1`,
`shape/1`, `dtype/1`, `eval/1`) remains available for diagnostics
and direct MLX round-trips, but most users should go through Nx.

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

**Stream lifecycle.** `Emily.Stream` has no explicit release API;
cleanup piggybacks on BEAM GC of the NIF resource held in the
struct. In the pattern above, the stream lives as long as its owning
worker process: when the worker terminates (crash, supervisor
shutdown, or `GenServer.stop/1`), the process heap is reclaimed, the
resource's refcount drops to zero, and the NIF destructor joins the
dedicated OS thread. A supervised restart therefore drops the old
stream and allocates a fresh one in the child's `init/1`. To drop a
stream deliberately, terminate the process that owns it.

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

See `Emily.Stream` for the API and the `qwen3_quantized` notebook
under Notebooks for a worked multi-stream example.

## Observability

Emily emits `:telemetry` events at the evaluation boundary
(`[:emily, :eval, *]`, `[:emily, :to_binary, *]`) and at every
`Nx.BinaryBackend` fallback (`[:emily, :fallback, *]`). Attach a
handler to graph hotspots or detect silent performance cliffs —
see `Emily.Telemetry` for the full event catalogue.

When a backend callback has no native MLX path, Emily transparently
falls back to `Nx.BinaryBackend`. The fallback is ~100× slower; to
get a one-shot `Logger.warning` per `{op, input_shapes}` pair the
first time each one fires (recommended during development):

```elixir
# config/dev.exs
config :emily, :warn_on_fallback, true
```

The warning is off by default so library consumers and CI logs stay
quiet. The telemetry event fires regardless.

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

## License

MIT
