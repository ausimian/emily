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
MLX C++           (vendored source)  — built from `vendor/mlx` via cmake
```

Dispatch is one-directional (Elixir → C++ → MLX); the C++ side never
calls back into the BEAM.

## Features

- **Nx backend.** Every `Nx.*` op dispatches to MLX; ops without a
  native primitive fall back transparently to `Nx.BinaryBackend`
  with a `[:emily, :fallback, *]` telemetry event. See
  `Emily.Backend`.
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

- **macOS.** Apple Silicon recommended; x86_64 is supported but
  without GPU acceleration.
- **Elixir 1.18+ / OTP 27+.** Development is pinned to Elixir 1.19.5
  / OTP 28.3 via `.tool-versions`.
- **Xcode with the Metal toolchain.** The Command Line Tools alone
  are not enough — the build invokes `xcrun -sdk macosx metal`,
  which is only reachable from a full Xcode install. From a fresh
  macOS:

  ```sh
  # 1. Install Xcode from the App Store, then point xcode-select at it:
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

  # 2. If Xcode is present but the Metal Toolchain component is missing:
  xcodebuild -downloadComponent MetalToolchain
  ```

  `mix compile` will surface the correct command in its error message
  if the toolchain is unreachable.
- **cmake.** Used to build MLX from the vendored source tree. Install
  via Homebrew (`brew install cmake`) or the equivalent for your
  setup.

## Building

MLX is vendored as a git submodule under `vendor/mlx` and built from
source during `mix compile`. There is no prebuilt download step.

### Why vendored?

The vendored pin sits ahead of the latest tagged MLX release because
Emily depends on two thread-safety changes that were merged to MLX
`main` after the last release:

- [ml-explore/mlx#3348](https://github.com/ml-explore/mlx/pull/3348) —
  thread-local `CommandEncoder`. Each MLX stream's Metal encoder lives
  on the thread that created it, which lets Emily pin a stream to a
  dedicated worker thread without colliding with other streams.
- [ml-explore/mlx#3405](https://github.com/ml-explore/mlx/pull/3405) —
  `ThreadLocalStream` API. Lets the worker thread set its stream as
  the per-thread default so MLX ops dispatched on that thread go to
  the right queue without explicit threading at every call site.

Before these landed, concurrent dispatch from multiple OS threads
would race the Metal driver (the conformance suite hit SIGABRTs and
SIGSEGVs when the BEAM migrated processes between schedulers). With
them, Emily can run one MLX stream per worker thread and let multiple
BEAM processes drive inference concurrently — see the
[Concurrency model](#concurrency-model) section below.

Once a tagged MLX release contains both PRs, Emily can switch to a
released version and drop the submodule. Until then, vendoring is the
cleanest way to pin a known-good `main` commit reproducibly.

### How to build

```sh
git clone --recurse-submodules https://github.com/ausimian/emily.git
cd emily
mix deps.get
mix compile
```

If you cloned without `--recurse-submodules`, initialise them:

```sh
git submodule update --init --recursive
```

The first build takes several minutes to compile MLX itself. The
artefact (`libmlx.a` + the Metal shader library `mlx.metallib`) is
cached under `$EMILY_CACHE` (default:
`~/Library/Caches/emily/mlx-<submodule-hash>`) and reused across
builds until the submodule pin changes. Override the cache location
with `EMILY_CACHE=/some/path mix compile`, or force a rebuild with
`mix compile.emily_mlx --force`.

### MLX JIT (optional)

MLX can ship its Metal kernels either AOT-compiled into
`mlx.metallib`, or as source strings that are JIT-compiled on first
use. `EMILY_MLX_JIT=1` at build time selects the JIT path; any other
value (or unset) is the default and produces the AOT build. Toggling
the flag produces a distinct cached MLX build (the cache key includes
the setting), so no stale artefact is reused.

Artefact sizes on an M-series Mac (dev build, release optimisations):

| Mode                       | `libemily.so` | `mlx.metallib` | `priv/` total |
| -------------------------- | ------------: | -------------: | ------------: |
| JIT off (default)          |       ~20 MB  |       ~154 MB  |      ~175 MB  |
| JIT on (`EMILY_MLX_JIT=1`) |       ~22 MB  |       ~3.5 MB  |       ~25 MB  |

With JIT on, kernels are compiled on first invocation, so there's a
small per-kernel warm-up cost at runtime; subsequent calls are cached
in-process. All of Emily's test suite passes under both modes.

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
that runs the MLX ops on behalf of BEAM schedulers. NIFs hand their
work to a worker via a `run_sync` promise (blocks the caller for
~1–10 µs) rather than executing on the scheduler thread directly,
which keeps MLX's per-thread `CommandEncoder` consistent and lets
the BEAM migrate Elixir processes freely.

By default, every op uses the **default worker** owned by the
`Emily.MlxStream.Default` GenServer under the application supervisor.
That single queue serialises all GPU work across the VM — correct
and simple, but a bottleneck under concurrent inference.

**Stream-per-process** — for concurrent inference on a shared model:

```elixir
stream = Emily.Stream.new(:gpu)

Emily.Stream.with_stream(stream, fn ->
  # Every Emily op in this block dispatches on `stream`'s
  # Metal command queue — concurrent with other streams.
  model.(input)
end)
```

Each `Emily.Stream` maps to its own `WorkerThread` and its own Metal
command queue. Weights are shared across streams (MLX arrays are
refcounted and thread-safe for reads), so the per-stream cost is the
command buffer, not the model. Create streams once at serving-worker
init, not per-request.

**Pooled servings** — for small models where duplicating weights is
cheap, start K `Nx.Serving` instances behind poolboy / Registry /
etc. Each instance holds its own weights and runs on the default
stream. No `Emily.Stream` needed. Trade-off: memory scales linearly
with K.

See `Emily.Stream` for details and the `qwen3_quantized` notebook
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
