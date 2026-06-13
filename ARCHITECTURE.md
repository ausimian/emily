# Emily — architecture

The current shape of the library, in a form that contributors can read
without sifting milestone history. For per-milestone rationale see
[`PLAN.md`](https://github.com/ausimian/emily/blob/main/PLAN.md); for open work see [`ROADMAP.md`](ROADMAP.md).

## Layers

```
Emily.Compiler    (Nx.Defn.Compiler) — validates opts, pins the result backend,
                                        selects one of two lowering lanes:
                                          • native (default, native: true):
                                            lower once to Emily.IR/Program,
                                            single-NIF replay
                                          • eval (native: false): op-by-op via
                                            Nx.Defn.Evaluator + Emily.Backend
Emily.IR + Program (native lane)     — flat IR compiled once; whole-graph replay,
                                        optionally mx::compile-fused (:fuse)
Emily.Backend     (Nx.Backend)       — op-by-op translation to Native
Emily.Native      (NIF shim)         — one function per MLX op, no policy
Worker threads    (one OS thread per stream)
MLX C++                              — statically linked into libemily; mlx.metallib alongside
```

Dispatch is unidirectional in the deadlock-class sense: every NIF
enqueues its work on a worker and returns immediately. The worker
posts its result back to the caller via `enif_send` — fire-and-forget,
no synchronous return value — and the Elixir wrapper awaits the
message with a plain `receive`. C++ never calls into Elixir code, and
no NIF ever blocks on a BEAM operation. Each layer has its own oracle
so a bug can only be introduced in the layer where its test fails —
see [Testing philosophy](#testing-philosophy).

## Core design decisions

1. **Backend-first; compiler layered on top.** The `Nx.Backend` is
   enough to run Bumblebee; the `Nx.Defn` compiler is additive. In
   both lanes `Emily.Compiler` pins the result backend to
   `Emily.Backend` via `__to_backend__/1` and caps `:max_concurrency`
   at 1. By default (`native: true`) it lowers to the single-NIF lane
   (decision 10); `native: false` delegates the expression walk to
   `Nx.Defn.Evaluator` op-by-op instead, and `:fuse` wraps
   `mx::compile` on top of the native lane.
   The original PLAN M6 measurement found whole-graph `mx::compile`
   below the 1.20× gate; the single-NIF replay changed that economics
   (fuse the elementwise runs the replay leaves separate, and cache a
   fused `defn while` body per stream), so fusion now ships as the
   opt-in `:fuse` mode.

2. **Trace in Elixir, not in C++.** `Nx.Defn.Expr` is already a
   fully traced tree. The default native lane lowers it to `Emily.IR`
   and emits a single NIF call for the whole graph; the op-by-op lane
   (`native: false`) walks it from Elixir and emits one Native NIF
   call per node (`lib/emily/native.ex`). Either way the trace is
   consumed in Elixir, never re-traced in C++.

3. **One resource type: `Tensor`** wrapping `mlx::array`. MLX's
   refcount does the heavy lifting; `fine`'s `ResourcePtr` adds one
   BEAM-managed ref. `c_src/emily/tensor.hpp` is the source of truth
   for the wrapper.

4. **Worker-thread dispatch.** Every NIF enqueues its work on a
   dedicated OS thread (the **worker**) that owns one MLX stream and
   its Metal command encoder. NIFs return immediately after
   enqueueing; the worker posts `{ref, {:ok, result}}` back via
   `enif_send` and the public Elixir wrapper awaits it with a plain
   `receive`. No BEAM scheduler (regular or dirty) blocks on MLX
   work. Because the MLX stream is pinned to its worker thread,
   Metal's per-thread `CommandEncoder` state stays consistent
   regardless of how the BEAM migrates Elixir processes between
   schedulers.

5. **Default worker + per-process streams.** The
   `Emily.MlxStream.Default` GenServer owns the default worker. Every
   op uses it unless the caller has installed a per-process worker
   via `Emily.Stream.with_stream/2`. Per-process workers are the
   recommended pattern for concurrent inference on a shared model —
   weights live in one MLX buffer, each worker reads from it
   independently.

6. **Cache compiled defn in the closure `Nx.Defn.compile/3` returns**,
   not in ETS or `:persistent_term`. Bumblebee and `Nx.Serving` hold
   that closure on warmup, so subsequent calls skip the walk. An
   external `{mfa, input_signature}` cache was prototyped and dropped
   — the per-call ETS deep-copy cost on a Qwen3-sized expression tree
   exceeded any reuse savings.

7. **No f64.** Hard error at the Backend with a clear message
   pointing to f32. MLX has no f64 primitive on Metal; not worth
   working around. Same goes for `{:f8_e4m3fn, 8}` (introduced in Nx
   0.11) — rejected at the boundary with an "no MLX primitive"
   `ArgumentError`.

8. **Error discipline.** Every NIF catches C++ exceptions at the
   boundary and returns `{:error, term}`. Never unwind across
   `enif_` calls. Async errors are annotated with op, input
   shapes/dtypes, options, and worker context — see the async
   helper module under `lib/emily/native/async.ex`.

9. **Zero-copy `to_binary`.** `Nx.to_binary/1` on an Emily tensor
   returns a BEAM resource binary aliasing the MLX buffer via
   `enif_make_resource_binary`; the resource retains a refcount on
   the `mlx::array` so the buffer survives until the BEAM binary is
   GC'd. `from_binary` retains its memcpy — `MTL::newBufferWithBytesNoCopy`
   requires page-aligned, page-sized memory that real-world inputs
   (safetensors, `:file.pread`) never provide.

10. **Native single-NIF compiler (`native: true`).** Lowers the
    traced `Nx.Defn.Expr` to a flat `Emily.IR` (`lib/emily/ir.ex`),
    compiles it once to an `Emily.Program` (`lib/emily/program.ex` +
    `c_src/program.cpp`), and replays the whole forward graph in a
    single NIF call per invocation — collapsing the per-op BEAM↔worker
    round-trips the default lane pays on a decode loop. Weights cross
    the NIF boundary once, captured by the compiled program. Anything
    the IR can't lower routes the *whole* defn back through
    `Nx.Defn.Evaluator` under the default `native_fallback: :eval`
    (firing `[:emily, :compiler, :fallback]`), so native is safe as
    the default on any model. Use `native_fallback: :raise` to fail
    instead — the conformance gates use it to prove full native
    lowering. The default comes from `config :emily, :native` (itself
    `true`); set `config :emily, native: false` (or pass
    `native: false`) to opt a memory-constrained host back to op-by-op. `:fuse` wraps the replay in `mx::compile`, cached per stream
    and fusing `defn while` bodies for decode loops — not
    bit-identical, hence opt-in.

## Concurrency model

MLX dispatches GPU work through Metal command queues. Emily owns one
worker thread per command queue. The default worker is shared across
the VM; per-process workers (`Emily.Stream`) let multiple processes
run inference concurrently on a shared model.

Three viable configurations for serving:

| Configuration                            | Weights | GPU queues | When to use                                              |
| ---------------------------------------- | ------- | ---------- | -------------------------------------------------------- |
| Single serving, default stream           | 1×      | 1 (shared) | Default. Simplest; fine for single-user / batched.       |
| Single serving + pool of `Emily.Stream`s | 1×      | N          | Concurrent inference on a shared model. Large models.    |
| K servings (pooled), default stream      | K×      | 1 (shared) | Small models where CPU serving work dominates GPU.       |

The README's _Concurrency model_ section has the worked code; this
note is for the architecture map only.

## Memory model

MLX buffers live outside the BEAM heap. `Emily.Memory` is the public
allocator API:

  * `stats/0` samples active / peak / cache bytes and emits the
    `[:emily, :memory, :stats]` telemetry event.
  * `reset_peak/0` resets the high-water mark.
  * `clear_cache/0` asks MLX to release cached reusable buffers —
    does **not** free live tensors. Tensors and resource binaries
    returned by `Nx.to_binary/1` are released only after the owning
    BEAM references are garbage collected.

`Emily.Telemetry.memory_stats/0` delegates here for back-compat; new
code should call `Emily.Memory` directly.

## Observability

Telemetry events — spans at the evaluation boundary, plus a few
discrete (non-span) events:

  * `[:emily, :eval, *]` — every `Emily.eval/1` and the implicit
    evaluation inside `to_binary`.
  * `[:emily, :to_binary, *]` — both `Emily.to_binary/1` and the
    `Nx.Backend.to_binary` path. Metadata: `:shape`, `:dtype`,
    `:byte_size`.
  * `[:emily, :fallback, *]` — every `Nx.BinaryBackend` fallback
    entry. Metadata: `:op`, `:input_shapes`, `:input_dtypes`.
  * `[:emily, :block, :fallback]` — discrete event each time the
    backend's `block` callback (Nx 0.12+ `Nx.Block.*` dispatch) falls
    through to the supplied default `fun`.
  * `[:emily, :compiler, :fallback]` — discrete event when a
    `native: true` defn can't be lowered and routes the whole defn
    through `Nx.Defn.Evaluator` instead. Metadata: `:key`, `:reason`
    (the lowering error, naming the unsupported op/construct).
  * `[:emily, :memory, :stats]` — discrete event from
    `Emily.Memory.stats/0`.

Span instrumentation deliberately stops at the evaluation boundary
rather than wrapping every graph-construction call site in
`Emily.Backend`: those NIFs are <10μs and do no work; the evaluation
boundary is where MLX actually runs kernels.

Fallback behaviour is configured via `config :emily, :fallback`:

  * `:silent` (default) — only the telemetry event fires.
  * `:warn` — one-shot `Logger.warning` per `{op, input_shapes}` pair.
  * `:raise` — `RuntimeError` on fallback entry; CI-friendly.

The legacy `config :emily, :warn_on_fallback, true` boolean is still
honoured when `:fallback` is unset.

## Debug assertions

Two compile-time opt-in flags re-enable runtime checks that MLX (and
every other GPU backend) skips by default. Both default `false`; the
guarded branches are dead-code eliminated by the Elixir compiler, so
the runtime cost when off is zero.

  * `:debug_bounds_check` — raises on out-of-range / negative indices
    in `gather` / `take` / `take_along_axis` / `indexed_add` /
    `indexed_put`.
  * `:debug_detect_nan_inf` — scans results of `matmul`, the fused
    `layer_norm` / `rms_norm`, and both fused SDPA variants.

Each check is a per-op MLX reduction plus a scalar readback — a
worker sync that breaks lazy-graph fusion. Leave off in release
builds.

## Build & packaging

  * **Hex consumers** download a precompiled NIF (`libemily.{so,dylib}`
    + `mlx.metallib`) from the GitHub release for the pinned version,
    SHA256-verified against a `.sha256` sidecar fetched alongside. No
    C++ toolchain or cmake required.
  * **Contributors** build from source: `mix deps.get` clones MLX into
    `deps/mlx_src` at the pinned tag, `scripts/build-mlx.sh`
    cmake-builds `libmlx.a` + `mlx.metallib` into
    `$EMILY_CACHE/mlx-<version>-<variant>/`, and `elixir_make` links
    the NIF against it.
  * **MLX variant** selection is via `config :emily, variant: :aot |
    :jit`. `:aot` is the default and works on macOS 14+; `:jit` ships
    smaller artefacts but requires macOS 26.2+ at runtime.
  * **`mix emily.doctor`** verifies the local install: host platform,
    active variant, native artefacts in `priv/`, NIF loadability, and
    a tiny `Emily.Backend` smoke test that asserts no silent fallback
    to `Nx.BinaryBackend`.

## Testing philosophy

Each layer is tested against its own oracle. A bug can only be
introduced in the layer where its test fails — no cross-layer
mystery bugs.

| Layer    | Oracle                                                   | Harness                                                    |
| -------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| Native   | Hand-computed expected values                            | ExUnit unit tests                                          |
| Backend  | `Nx.BinaryBackend` on the same inputs                    | StreamData property tests + Nx conformance                 |
| Compiler | `Emily.Backend` in non-defn mode                         | Equivalence tests (same function, two modes)               |
| Grad     | `Nx.BinaryBackend` grad + finite differences + EXLA CPU  | StreamData property tests + numerical oracle + EXLA golden |
| Training | `Nx.BinaryBackend` loss trajectory                       | Curve-matching; MNIST convergence (`:training_full`)       |
| E2E      | HuggingFace Transformers reference slices                | Bumblebee conformance suites with cached weights           |

Soak harnesses (all under `test/soak/`, `@tag :soak`, opt-in):

  * `memory_test` — 10k iterations; MLX memory returns to baseline.
  * `training_test` — 1k training steps; baseline restored after
    `Emily.Memory.clear_cache/0`.
  * `backend_concurrency_test` / `eval_concurrency_test` /
    `stream_concurrency_test` — parallel inference under the
    default worker, the evaluation path, and per-process streams;
    determinism + no crashes.
  * `backend_soak_test` — broad backend exerciser; allocator
    drift over a large mixed-op workload.
  * `quantized_memory_test` — quantized-matmul loop, distinct
    allocator pattern from fp16 inference.
  * `zero_copy_roundtrip_test` — `to_binary` aliases the MLX buffer
    rather than copying; tested via allocator-stats deltas.

## Risks and mitigations

| Risk                                                            | Mitigation                                                                          |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| MLX op semantics drift from Nx expectations                     | Property tests explicitly generate edge cases; document intentional divergences     |
| Metal driver bugs in specific macOS versions                    | Pin known-good macOS in CI; test matrix across 14/26                                |
| f16/bf16 accumulation differences from EXLA                     | Tolerance-aware comparisons; document expected divergence                           |
| Upstream Nx API changes (`Nx.Backend` / `Nx.Defn.Compiler` etc) | Version-pin Nx; coordinate with elixir-nx maintainers                               |
| MLX upstream API churn on source builds                         | `@mlx_version` pin; audit on bump; `mix emily.doctor` surfaces toolchain mismatches |
