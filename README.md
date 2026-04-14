# emily

Elixir bindings and Nx backend for Apple's [MLX](https://github.com/ml-explore/mlx).

**Status: M7 — Bumblebee conformance breadth.** Backend (M2), Defn
compiler (M5), and four Bumblebee models (DistilBERT, Qwen3, ViT,
Whisper) run end-to-end. M8 (native `conv`) and M9 (1.0 release) are
next. See [`PLAN.md`](PLAN.md) for the full roadmap and
[`RELEASE.md`](RELEASE.md) for unreleased-version notes.

## Why

To run Bumblebee models (notably Qwen3) on Apple Silicon with Metal
acceleration, via a layered architecture that keeps each layer
independently testable and avoids the [nif_call-deadlock
class](https://github.com/elixir-nx/emlx/issues/88) that grounded EMLX.

## Architecture

```
Emily.Compiler    (Nx.Defn.Compiler) — validates opts, pins the result backend
Emily.Backend     (Nx.Backend)       — op-by-op translation to Native
Emily.Native      (thin NIF shim)    — one function per MLX op, no policy
MLX C++           (vendored binary)  — cocoa-xu/mlx-build prebuilts, pinned
```

One-directional dispatch: Elixir → C++ → MLX. C++ never calls back into
BEAM, so the EMLX #88 deadlock class is structurally impossible.

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
HuggingFace Transformers (or EXLA) reference slices end-to-end. A bug
can only be introduced in the layer where its test fails.

## License

MIT
