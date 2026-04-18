# emily

Elixir bindings and Nx backend for Apple's [MLX](https://github.com/ml-explore/mlx).

**Status: M14 — Serving concurrency.** Stream-per-process
concurrent inference via `Emily.Stream`. See [`PLAN.md`](PLAN.md) for
the full roadmap and [`RELEASE.md`](RELEASE.md) for unreleased-version
notes.

## Why

To run Bumblebee models (notably Qwen3) on Apple Silicon with Metal
acceleration, via a layered architecture that keeps each layer
independently testable.

## Architecture

```
Emily.Compiler    (Nx.Defn.Compiler) — validates opts, pins the result backend
Emily.Backend     (Nx.Backend)       — op-by-op translation to Native
Emily.Native      (thin NIF shim)    — one function per MLX op, no policy
MLX C++           (vendored binary)  — cocoa-xu/mlx-build prebuilts, pinned
```

One-directional dispatch: Elixir → C++ → MLX. C++ never calls back
into BEAM.

## Requirements

- macOS (Apple Silicon recommended; x86_64 supported)
- Elixir 1.18+ / OTP 27+ (development pinned to 1.19.5 / OTP 28 via `.tool-versions`)

MLX 0.25.1 is fetched as a prebuilt from
[cocoa-xu/mlx-build](https://github.com/cocoa-xu/mlx-build) during
`mix compile`; no separate install step.

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
Whisper pipelines.

The low-level tensor API (`Emily.from_binary/3`, `to_binary/1`,
`shape/1`, `dtype/1`, `eval/1`) remains available for diagnostics and
direct MLX round-trips, but most users should go through Nx.

## Concurrency

MLX dispatches GPU work through Metal command queues. By default all
ops share one queue (the default stream), which is not safe for
concurrent dispatch from multiple OS threads.

**Stream-per-process** — for concurrent inference on a shared model:

```elixir
stream = Emily.Stream.new(:gpu)

Emily.Stream.with_stream(stream, fn ->
  # All Emily ops here dispatch on this stream's command queue.
  model.(input)
end)
```

Each stream maps to its own Metal command queue. Multiple processes
can run inference concurrently — one shared model, no weight
duplication. Create streams at init time (one per serving process),
not per-request.

**Pooled servings** — for simpler setups with small models, start K
`Nx.Serving` instances behind a pool (poolboy, Registry, etc.). Each
instance loads its own weights and runs on the default stream. No
`Emily.Stream` needed. Trade-off: each pool member holds its own
weight copy.

See `Emily.Stream` moduledoc for details.

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

## Milestones shipped

- **M0** — NIF scaffold, MLX prebuilt fetch, tensor round-trip.
- **M1** — `Emily.Native` op inventory (creation, unary, binary,
  reductions, shape, indexing, sort, linalg, FFT, random, memory).
- **M2** — `Emily.Backend` (`Nx.Backend`). StreamData property oracle
  vs. `Nx.BinaryBackend`; soak + concurrency harnesses.
- **M3** — DistilBERT end-to-end on Bumblebee; native batched `dot`,
  type promotion, `bitcast`.
- **M4** — Qwen3 (`Qwen/Qwen3-0.6B`) greedy decode end-to-end; native
  `put_slice` for KV-cache.
- **M5** — `Emily.Compiler` (`Nx.Defn.Compiler`): validates opts, pins
  the result backend, delegates the walk to `Nx.Defn.Evaluator`.
- **M6** — **dropped** after Phase-1 de-risk.
  `mlx::core::compile` wrapping measured <1.20× on transformer-shaped
  workloads (regression on CPU) and was cut. Microbench harness
  retained at `bench/native/compile_microbench.cpp` /
  `mix bench.native` so the decision can be re-measured against
  future MLX releases. Full results:
  [`bench/compile_microbench.md`](bench/compile_microbench.md).
- **M7** — Bumblebee conformance breadth. ViT
  (`google/vit-base-patch16-224`) and Whisper (`openai/whisper-tiny`)
  each ship tiny-random (`@moduletag :conformance`) and
  full-checkpoint (`@moduletag :vit_full` / `:whisper_full`) tiers.
  `mix test --only conformance` now aggregates 14 tiny-random tests
  across DistilBERT, Qwen3, ViT, and Whisper.

## Testing

```bash
mix test                           # fast suite (unit + property)
mix test --only conformance        # + Bumblebee tiny-random suites
mix test --only qwen3_full         # full Qwen3-0.6B checkpoint (~1.5 GB)
mix test --only vit_full           # full ViT-base (~330 MB)
mix test --only whisper_full       # full whisper-tiny (~150 MB)
mix test --only soak               # memory + concurrency soak harnesses
```

Each layer has its own oracle: hand-computed expected values at the
Native layer, `Nx.BinaryBackend` on the same inputs at the Backend
layer, `Emily.Backend` in non-defn mode at the Compiler layer, and
HuggingFace Transformers reference slices end-to-end. A bug can only
be introduced in the layer where its test fails.

## License

MIT
