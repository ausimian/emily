# emily — implementation plan

Living document. Update per milestone; keep the rationale alongside the
checklist so future-us understands the trade-offs.

## Goals

- **Correctness over performance at every layer.** A fast library that
  produces wrong outputs is worthless. Every layer has its own oracle
  and its own test suite.
- **Structural impossibility of the [EMLX #88](https://github.com/elixir-nx/emlx/issues/88)
  deadlock class.** No bidirectional NIF calls. No GenServer on the hot
  path. No closures shipped into `Task.async` during graph
  construction.
- **Bumblebee-first.** DistilBERT → Qwen3 → Qwen3-VL as the canonical
  integration targets.
- **Shippable at every milestone.** Backend-only mode is useful on its
  own; the Defn compiler is additive.

## Non-goals (v1)

- Ahead-of-time compilation (IREE-style). Complementary, separate effort.
- Windows or non-Apple-Silicon Linux GPU. CPU-only Linux is a nice-to-have for CI.
- Training / gradients beyond what `Nx.Defn` gives for free. Inference is the priority.
- Drop-in replacement for EMLX. We borrow where it's clearly right, but
  we're not constrained by its API.
- `Emily.Stream` as a public API — MLX streams stay internal in v1.

## Architecture

```
Emily.Compiler    (Nx.Defn.Compiler) — optional, walks Nx.Defn.Expr
Emily.Backend     (Nx.Backend)       — op-by-op translation to Native
Emily.Native      (thin NIF shim)    — one function per MLX op, no policy
MLX C++           (vendored binary)
```

One-directional dispatch only: Elixir → C++ → MLX. C++ never calls back
into BEAM.

## Core design decisions

1. **Backend-first; compiler layered on top.** The Backend is enough to
   run Bumblebee. `mlx::core::compile` is an opt-in optimisation added
   last.
2. **Trace in Elixir, not in C++.** `Nx.Defn.Expr` is already a fully
   traced tree; we walk it from Elixir and emit one `Emily.Native` call
   per node. No C++→BEAM callbacks.
3. **`fine` for NIF ergonomics.** C++17 NIFs via
   [elixir-nx/fine](https://github.com/elixir-nx/fine) — auto-encoding,
   clean resource handling, dirty-scheduler flag per NIF.
4. **Vendor MLX via cocoa-xu's prebuilts.** Same source EMLX uses. Pin
   a specific version; upgrade deliberately.
5. **One resource type: `Tensor`** wrapping `mlx::array`. MLX's refcount
   does the heavy lifting; fine's `ResourcePtr` adds one BEAM-managed
   ref.
6. **Scheduler policy:**
   - Graph-construction NIFs (lazy ops): regular scheduler, <10μs each.
   - Materialisation (`eval`, `to_binary`, `item`): dirty CPU.
7. **Minimal supervisor tree.** Empty in M0; future: memory/stats agent.
   Dispatch never goes through a GenServer.
8. **Cache compiled defn in ETS**, keyed by `{mfa, input_signature}`.
   Not `:persistent_term` (expensive writes, GCs readers). Not a
   GenServer (bottleneck).
9. **Unified-memory zero-copy.** On Apple Silicon: `Nx.from_binary` can
   wrap the binary pointer; `to_binary` views the CPU-addressable MTL
   buffer. Benchmark-verified before claimed.
10. **No f64.** Hard error at the Backend with a clear message pointing
    to f32. Metal limitation; not worth working around.
11. **Error discipline.** Every NIF catches C++ exceptions at the
    boundary and returns `{:error, term}`. Never unwind across
    `enif_` calls.

## Milestones

### M0 — Scaffold

- `mix new emily` with library conventions
- `elixir_make` + Makefile; `cocoa-xu/mlx-build` prebuilt fetch
- Minimal NIF: `Tensor` resource + `from_binary`/`to_binary`/`shape`/`dtype`/`eval`
- Smoke tests that round-trip tensors across multiple dtypes

**Exit:** `mix test` passes; `Emily.from_binary(bin, shape, dtype) |> Emily.to_binary() == bin`.

### M1 — `Emily.Native`: complete op inventory

- Port the MLX op surface to NIFs, organised by file (`ops/creation.cpp`, `ops/binary.cpp`, ...)
- Each NIF has an ExUnit test that calls it directly with hand-computed expected outputs
- Resource-lifecycle soak test: allocate/drop, assert MLX memory stats return to baseline
- Dtype × op smoke matrix

**Testing — Layer 1 (Native):** unit tests, not property tests; we're
testing the shim, not the maths. Stress test for memory leaks. Error-path
tests (wrong dtype/rank/axis).

**Exit:** every MLX op we care about is callable from Elixir with
correct outputs and no leaks.

### M2 — `Emily.Backend`: the `Nx.Backend` implementation

- Implement `Nx.Backend` callbacks, each delegating to `Emily.Native`
- Zero-copy `from_binary`/`to_binary` on Apple Silicon
- Backend-transfer implementations for `Nx.BinaryBackend` interop

**Testing — Layer 2 (Backend):** this is where we spend the most effort.

1. **Property-based oracle tests (StreamData)** — for every backend
   callback, generate random shapes/dtypes/values; assert output matches
   `Nx.BinaryBackend` within dtype-appropriate tolerance (ulp-based for
   floats, exact for ints).
2. **Nx conformance tests** — replicate Nx's own backend test suite.
3. **Soak tests** — 10k forward passes of a small model; assert memory
   returns to baseline after `Emily.Memory.clear_cache/0`.
4. **Concurrency tests** — 16 parallel processes, same computation,
   bit-for-bit identical outputs, no crashes.

**Exit:** Emily.Backend passes the oracle suite and Nx's own backend
tests.

### M3 — Bumblebee: DistilBERT end-to-end

- `Nx.global_default_backend({Emily.Backend, device: :gpu})`
- Load `distilbert-base-uncased`, run question answering
- Conformance test: golden logits checked in (produced by EXLA on Linux+CUDA)

**Exit:** `mix test --only conformance` passes; `Nx.Serving.batched_run`
works with DistilBERT under load.

### M4 — Qwen3 inference

- Run `Qwen/Qwen3-0.6B-Instruct` end-to-end via Bumblebee causal-LM serving
- Golden-output test: fixed prompt, greedy decode, first 32 tokens match a checked-in reference
- Benchmark: tokens/sec on M2/M3/M4 Mac Mini hardware, vs cocoa-xu's
  pure-Elixir MLX harness as ceiling

**Exit:** Qwen3 produces correct output and we have a tracked
throughput number.

### M5 — `Emily.Compiler`: `Nx.Defn.Compiler` implementation

- Walk `Nx.Defn.Expr` in Elixir, dispatching each node to `Emily.Backend`.
  In practice this is what `Nx.Defn.Evaluator` already does — it dispatches
  via `Nx.Shared.list_impl!/1` which finds whichever backend the operands
  carry. `Emily.Compiler` validates options, points `__to_backend__/1` at
  `Emily.Backend`, pins partitions to 1 (MLX kernel dispatch isn't
  thread-safe), and delegates the walk.
- Hold the walked plan in the closure returned by `__compile__/4`; the
  closure *is* the cache. Callers that want reuse across invocations use
  `Nx.Defn.compile/3` and hold the returned function — Bumblebee /
  `Nx.Serving` already do this on warmup.
  - *Earlier draft proposed an ETS cache keyed by `{mfa, input_signature}`;
    rejected once we accounted for the per-call ETS deep-copy cost on a
    Qwen3-sized expression tree. The closure-capture path avoids the copy
    and matches the upstream Evaluator pattern.*
- **Do not use `mlx::core::compile` yet.** Lazy eval at the Backend layer suffices.

**Testing — Layer 3 (Compiler):**

- **Equivalence tests**: a representative sample of ops (creation,
  binary, reduction, shape, dot, container output) plus the `defn`-only
  constructs `while` and `cond`; assert `compiler: Emily.Compiler` matches
  raw Backend execution (and `Nx.Defn.Evaluator` for the `defn`-only
  cases). The full Backend property suite isn't re-run per op — the
  Backend already passes its own oracle suite, and the Compiler test
  is structural ("did the walk reach the right backend with the right
  args").
- **Reuse**: a `Nx.Defn.compile/3` closure runs many inputs of the same
  signature without re-walking the expression.
- **Callback contracts**: `__to_backend__`, `__partitions_options__`,
  unknown-option rejection, `:max_concurrency > 1` refusal.

**Exit:** Axon MLPs forward with `compiler: Emily.Compiler`; results
match `Nx.Defn.Evaluator` running on the same backend within float
tolerance. (Training is out of scope for v1.)

### M6 — `mlx::core::compile` wrapping

- After Compiler has built the lazy op sequence, optionally wrap it in
  `mlx::core::compile` for shape-pinned specialisation
- Purely an optimisation; Backend-only path remains the default
- Thunk constructed in a single NIF call from a recorded op list; MLX
  traces a closure that replays ops against placeholders — no BEAM
  callbacks

**Testing:** equivalence tests rerun with `mlx_compile: true`; benchmark
speedup on a Qwen3 forward pass. If <20% win, drop.

**Exit:** compile mode is off by default, opt-in, demonstrably faster,
zero regressions.

### M7 — 1.0 release

- API docs, HexDocs, README with a worked Bumblebee example
- Hex release (public), versioned per conventions (`@version` in mix.exs)
- `RELEASE.md` accumulated across feature branches

## Testing philosophy

| Layer | Oracle | Harness |
|---|---|---|
| Native | Hand-computed expected values | ExUnit unit tests |
| Backend | `Nx.BinaryBackend` on the same inputs | StreamData property tests + Nx conformance |
| Compiler | `Emily.Backend` in non-defn mode | Equivalence tests (same function, two modes) |
| E2E | EXLA-produced golden outputs | Conformance tests with cached weights |

A bug can only be introduced in the layer where its test fails — no
cross-layer mystery bugs.

Additional harnesses:
- **Memory soak** (`test/soak/memory_test.exs`, `@tag :soak`): 10k
  iterations; MLX memory stats asserted to return to baseline.
- **Concurrency** (`test/soak/concurrency_test.exs`, `@tag :soak`):
  parallel inference; determinism + no crashes.
- **Benchmarks** (`bench/`): Benchee scripts; results logged in
  `RELEASE.md` per version.
- **Conformance vs EXLA** (CI matrix, Mac for Emily + Linux+CUDA for
  EXLA oracle): same model, same input; runs on every PR touching
  Backend.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| MLX op semantics drift from Nx expectations (NaN handling, int overflow, sort stability) | Property tests explicitly generate edge cases; document intentional divergences |
| Zero-copy assumption breaks on future hardware | Benchmark the copy; fall back cleanly; don't depend on zero-copy for correctness |
| `mlx-build` prebuilts stall or go unmaintained | Have a source-build fallback (env: `EMILY_BUILD_MLX=true`); CI job tests it monthly |
| Metal driver bugs in specific macOS versions | Pin known-good macOS in CI; test matrix across 14/15/26 |
| f16/bf16 accumulation differences from EXLA | Tolerance-aware comparisons; document expected divergence |
| Upstream Nx API changes (Defn.Compiler internals not stable) | Version-pin Nx; coordinate with elixir-nx maintainers |

## Project decisions (ratified)

- **Repo**: GitHub under `ausimian`. Push deferred.
- **Publishing**: public `hex.pm`. Deferred.
- **Streams**: internal only in v1.
- **Training**: out of scope for v1 (inference only).
- **EMLX coordination**: none — quiet ship.
