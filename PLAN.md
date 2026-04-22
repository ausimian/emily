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
- Training framework features beyond `Nx.Defn.grad`: distributed training,
  a native optimizer library. (Autodiff + small-scale training loops: in
  scope from M9; mixed-precision master weights: in scope from M16.)
- Drop-in replacement for EMLX. We borrow where it's clearly right, but
  we're not constrained by its API.
- `Emily.Stream` beyond the narrow `with_stream/2` + `new/1` +
  `synchronize/1` surface M14 introduces for the documented
  "big model, multi-process serving" pattern.

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
   run Bumblebee. Wrapping `mlx::core::compile` was planned as an
   opt-in optimisation on top, but was dropped after de-risking — see
   M6 for the measurement and reasoning.
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
   buffer. Benchmark-verified before claimed. *Status: currently
   unimplemented — the v1 round-trip path memcpys unconditionally
   (`emily_nif.cpp:57-58, :74-84`). M12 delivers the zero-copy version
   and the verifying benchmark.*
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
- **Do not use `mlx::core::compile`.** M6 de-risked this and dropped
  it — the fusion win on transformer-shaped workloads is below the
  1.20× gate. Lazy eval at the Backend layer is the shipping story.

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
tolerance. (Training via `Nx.Defn.grad` lands in M9.)

### M6 — `mlx::core::compile` wrapping — **dropped**

De-risked in pure C++ before paying the Backend/Compiler integration
cost, per the PLAN gate ("If <20% win, drop"). Full results:
[`bench/compile_microbench.md`](bench/compile_microbench.md).

Summary of findings on MLX 0.25.1, Apple Silicon:

- **Pure elementwise workload (harness validation):** 2.78× on GPU,
  1.47× on CPU — confirms `mx::compile` does what it advertises when
  fusion is available.
- **Transformer block (Qwen3-0.6B-shaped, seq ∈ {128, 512}):**
  1.04–1.07× on GPU, **regression** (0.82–0.88×) on CPU. Fails the
  1.20× gate across every workload shape tested.

Why: transformer inference is matmul-dominated, and MLX's compile does
not fuse matmul kernels with adjacent elementwise ops. The fusion
surface (RMSNorm chains, softmax neighbourhood, SwiGLU's silu×up) is a
small fraction of block runtime, bounding whole-block speedup to
single-digit percent. On CPU the tape-replay overhead exceeds the
fusion gain.

The BEAM-integrated compile path could not outperform this C++ ceiling,
so shipping M6 would deliver a <20% speedup at best — and a regression
at worst if a user selects the CPU device.

**Artefacts retained** so the decision can be re-measured against
future MLX releases without rebuilding the harness:

- `bench/native/compile_microbench.cpp` — the microbench source
- `lib/mix/tasks/bench.native.ex` — `mix bench.native` task
- `bench-native` target in the root `Makefile`
- `bench/compile_microbench.md` — results + reproduction instructions

If MLX gains matmul-adjacent fusion (bias-fused matmul, attention
fusion outside `fast::scaled_dot_product_attention`), re-run the bench
and revisit.

**Re-measured 2026-04-22 on MLX 0.31.1+69 — decision stands.** GPU
transformer-block speedup rose to 1.09–1.11× median (from 1.04–1.07×
on 0.25.1); CPU is still a regression at 0.82–0.88×. Both still fail
the 1.20× gate. See the "MLX 0.31.1+69" section in
[`bench/compile_microbench.md`](bench/compile_microbench.md).

### M7 — Bumblebee conformance breadth: ViT + Whisper

DistilBERT (M3) and Qwen3 (M4) cover encoder-only and decoder-only
transformers but leave three architectural shapes untested: 2-D
convolution, encoder-decoder cross-attention, and the Bumblebee
vision/audio pipelines. M7 closes the first two gaps.

- **ViT** (`google/vit-base-patch16-224`): vision, encoder-only,
  conv patch embedding, GELU FFN, classifier head. First suite to
  exercise the `conv` fallback in anger (`lib/emily/backend.ex` still
  routes `conv` through BinaryBackend as of M7 — correct but slow).
- **Whisper** (`openai/whisper-tiny`): audio, encoder-decoder, 1-D
  conv encoder frontend, sinusoidal position encodings, and
  cross-attention KV-cache in the decoder.

Each suite ships two tiers: a tiny-random tier that mirrors
Bumblebee's own test (HuggingFace Transformers reference slices) and
a full-checkpoint tier with deterministic synthetic inputs pinned
against the real-weight forward pass on Emily. Both gated as in M3
and M4: `:conformance` for tiny (opt in via `--only conformance`),
per-model `:*_full` tag for full (opt in separately).

Shared scaffolding (`test/support/conformance_helper.ex`) lifts the
`setup_all` backend swap and `assert_all_close/3` out of each suite.

**MoE / Mixtral deferred**: the pinned Bumblebee ref ships no
Mixtral or MoE architecture. Track as a follow-up; revisit when
upstream lands.

**Exit:** ViT and Whisper each pass both tiers on Apple Silicon;
`mix test --only conformance` aggregates 14 tiny-random tests across
all four Bumblebee models.

### M8 — Native conv

Lift `Backend.conv` onto `Native.conv_general` (the NIF already
exists; only the Backend callback still routes through the
BinaryBackend fallback). Gated on the M7 ViT and Whisper suites
staying green through the switchover.

### M9 — Gradient conformance and training primitives

Training on Emily has been technically possible since M2 —
`Nx.Defn.grad` is pure symbolic differentiation in Elixir and lowers
to the same ops the forward pass uses. M9 turns "possible" into
"usable" by (a) lifting the training-hot indexing ops off the
`via_binary` fallback and (b) building the test scaffolding needed
to trust a gradient.

**Primitives.** `Nx.Defn.grad` of indexing-shaped ops lands on
`indexed_add`; every such backward currently ships to BinaryBackend
and back. Lift to native MLX:

- `indexed_add` → `mlx::core::scatter_add`
- `indexed_put` → `mlx::core::scatter`
- `gather` → `mlx::core::gather`

Window reductions stay on `via_binary` in M9 — pool-based conv
training is scoped to M17.

**Testing — Layers 4 (Grad) and 5 (Training):**

1. **Grad-equivalence property tests** — for a zoo of `defn`-expressible
   functions f, assert `Nx.Defn.grad(f)` on `Emily.Backend` matches the
   same grad on `Nx.BinaryBackend` within dtype-appropriate tolerance.
   Reuses M2's StreamData harness; the zoo excludes non-differentiable
   ops (`argmax`, `argmin`, `floor`, `sign`, comparisons).
2. **Numerical finite-difference oracle** — for the differentiable
   subset, assert `(f(x+ε) - f(x-ε)) / 2ε ≈ grad(f)(x)`. Tolerance is
   per-op and documented; f32 central differences bottom out around
   1e-3 relative, so symbolic-grad tolerance must be relaxed
   accordingly where this is the oracle. Pilot on 3–4 ops before
   scaling the harness.
3. **Training curve-matching** — handwritten MLP and handwritten
   transformer-block training step, fixed seed, 50–200 steps; assert
   per-step loss trajectory matches `Nx.BinaryBackend` within
   tolerance. No Axon dependency in this tier — fewer moving parts
   when a test goes red.
4. **Training memory soak** (`test/soak/training_test.exs`,
   `@tag :soak`) — 1k training steps; MLX memory returns to baseline
   after `clear_cache/0`. Training exercises a different allocator
   pattern than inference (param-grad pairs, optimizer state,
   long-lived activation caches).
5. **`:training_full`** (opt-in via `--only training_full`, **not**
   on default CI) — Axon MLP on MNIST → >97% test accuracy. Catches
   systemic numerical drift that curve-matching misses because both
   sides use `Nx.BinaryBackend` as the oracle.

Axon is added as a **test-only** dependency, used only by the
`:training_full` tier.

**Risks specific to this milestone:**

- f32 tolerance calibration for oracle (2) is per-op; the harness
  must support per-op tolerance tables, not a single global epsilon.
- Random-key flow through `Emily.Compiler` needs an explicit test:
  grad through `dropout` with threaded keys, two invocations of the
  same compiled function must advance the RNG correctly.
- MLX scatter semantics (out-of-bounds handling, tie-breaking) may
  differ from Nx expectations. Document divergence; encode property
  exclusions if needed.

**Exit:** oracles (1)–(3) green in default CI; (4) and (5) green in
opt-in CI job.

### Post-M9 priority ordering

The milestones below were derived from a structured review of the
post-M9 codebase (see PR discussion) and ordered by user-visible
value, not difficulty. Headline rationale:

1. **Quantization, fast kernels, zero-copy** (M10–M12) move the needle
   most for the headline use case (Bumblebee Qwen3 on a MacBook).
2. **Grad oracle, serving, linalg** (M13–M15) close correctness and
   production-readiness gaps that block real adoption.
3. **Mixed-precision and conv-pool training** (M16–M17) finish the
   training story M9 started.
4. **Observability, errors, interop, doctor** (M18–M21) are the polish
   that turns a working library into one new users can actually adopt
   without hand-holding.
5. **Debug flags and documentation** (M22–M23) close the last
   pre-release gaps: caller-error detection and the HexDocs / worked-
   example surface new users actually land on.
6. **1.0 release** (M24) ships the result.

### M10 — Quantized inference primitives (partial)

Quantization is the single largest gap between Emily and "actually run
Qwen3 on a 16 GB MacBook". MLX ships native int4/int8 affine
quantization (`mx::quantize`, `mx::dequantize`,
`mx::quantized_matmul`); M10 binds it at the Native and Elixir levels
and ships a direct-call helper for eager use. The Bumblebee-integrated
conformance path is split out to M10.5 — see **Scope note** below.

**Shipped**:

- **Native bindings**: `Native.quantize/3`, `Native.dequantize/5`,
  `Native.quantized_matmul/7` over the MLX C++ functions. `quantize`
  returns a 3-tuple `{w_q, scales, biases}`. `quantized_matmul/7` takes
  `transpose` as an explicit boolean rather than PLAN's original `/6`:
  AWQ packed layouts need `transpose=false` while fresh-from-dense
  weights use `transpose=true`, and MLX exposes it as a required
  parameter.
- **`Emily.QuantizedWeight`** (`lib/emily/quantized_weight.ex`) —
  `Nx.Container`-derived struct with `{value, scales, biases,
  group_size, bits, transpose}`. Scalar metadata survives container
  traversal via `Nx.Container`'s `keep:` option. `from_dense/2`
  validates rank, last-axis divisibility, dtype, and bit count.
- **`Emily.Quantization.quantized_matmul/2`** — direct-call helper that
  extracts refs from an input tensor and a `%QuantizedWeight{}` and
  dispatches the fused kernel. Intended for eager/benchmark use and as
  the substrate for M10.5's defn-integration path.
- **Memory soak** (`test/soak/quantized_memory_test.exs`) — 1000-iter
  quantized-matmul loop asserts active memory returns within 4 MB of
  baseline after `Native.clear_cache/0`. Kept separate from the fp16
  memory soak because quantized inference is allocator-pattern
  different: packed weights load once and never re-quantize.
- **Native unit tests + Backend property tests** — see
  `test/emily/native_test.exs` (+7 cases) and
  `test/emily/quantization/` (two new files). Round-trip `quantize →
  dequantize` and `quantized_matmul` vs. `matmul(x, dequantize(…))`
  oracles for both `transpose=true` and `transpose=false` layouts.

**Scope note — why no Backend routing / Axon integration / conformance
test in M10**:

- **`Backend.dot/7` dispatch doesn't work.** PLAN'd approach was
  "detect quantized operand structs at the Nx layer (Bumblebee tags
  them in the parameter map) and dispatch the matmul callback to
  `Native.quantized_matmul`". But `Nx.dot/2` calls
  `Nx.LazyContainer.traverse/3` expecting a single `%T{}`; a
  three-tensor `%QuantizedWeight{}` container raises before reaching
  `Backend.dot/7`.
- **Axon layer-op dispatch doesn't work either.** `Axon.layer` ops run
  at `Nx.Defn.jit` trace time with `Nx.Defn.Expr` inputs;
  `Nx.Defn.Evaluator` walks those expressions dispatching `Nx.Backend`
  callbacks with materialized refs. There is no public hook to inject
  a custom op like `Native.quantized_matmul` that isn't already a
  `Nx.Backend` callback, and `deftransform` / `hook` / metadata all
  run at trace time (no refs available).
- **Bumblebee has no AWQ loader yet.** The exploration for M10
  confirmed `deps/bumblebee` has zero quantization-loading code (no
  AWQ, GPTQ, MLX-format paths). PLAN.md's "Bumblebee can already load
  quantized checkpoints" is aspirational.

All three of these are meaningful scope. M10 ships the substrate they
all need; M10.5 picks the defn-integration strategy and ships the
conformance test.

**Exit**: Native NIFs green under unit + property tests; QuantizedWeight
container property tests green; direct-call helper green under oracle
comparison; quantized memory soak clean.

### M10.5 — Bumblebee quantized inference integration

Closes the gap M10 left open: `Native.quantized_matmul` is now
reachable from `Nx.Defn.jit`-traced Axon forward passes, and a
quantized Qwen3-0.6B greedy-decodes end-to-end under Bumblebee's
standard `Bumblebee.Text.generation/4` serving.

**Shipped**:

- **`Emily.Quantization.dequantize_defn/1`** (`lib/emily/quantization.ex`)
  — defn-native analogue of `QuantizedWeight.to_dense/1`, built from
  `Nx.right_shift` / `Nx.bitwise_and` / multiply / add. Supports
  `bits ∈ {2, 4, 8}` (lanes-per-u32 is integral). `bits ∈ {3, 6}`
  (cross-u32 packing) is out of scope — the Native path remains for
  those.
- **`Emily.Quantization.Layers.quantized_dense/4`**
  (`lib/emily/quantization/layers.ex`) — Axon-compatible layer op
  (`deftransform` → `defnp`). Pattern-matches on `%QuantizedWeight{}`,
  dispatches `Nx.dot(x, dequantize_defn(qw))` (`transpose=false`) or
  `Nx.dot(x, Nx.transpose(dequantize_defn(qw)))` (`transpose=true`).
  Two kernel dispatches per matmul instead of MLX's fused one —
  accepted trade-off for integration without forking
  `Nx.Defn.Evaluator`. M11's fast-kernel work closes the gap.
- **`Emily.Quantization.Transform`** (`test/support/quantization_transform.ex`)
  — graph rewriter + model-state quantizer, modeled on
  `Axon.Quantization`. `quantize/3` takes a dense Axon model + dense
  `Axon.ModelState` and returns the pair with every `:dense` node
  rewritten to `:quantized_dense` and every dense kernel replaced
  with `%QuantizedWeight{}`. Lives in `test/support/` because Axon is
  an `only: :test` dep of Emily; graduates to `lib/` when an upstream
  Bumblebee AWQ-loading path lands.
- **`:qwen3_quant_full` conformance test** (`test/emily/conformance/qwen3_quant_full_test.exs`)
  — loads dense Qwen3-0.6B via Bumblebee, quantizes via `Transform.quantize/3`
  (`bits=4, group_size=128, transpose=true`), runs
  `Bumblebee.Text.generation/4` greedy decode for 32 tokens. Pins the
  continuation string as a regression gate. Opt-in via `mix test --only
  qwen3_quant_full` (mirrors `:qwen3_full`'s model-size discipline).

**Approach chosen**: Option 1 (defn-native dequantize). Option 2
(`Emily.Compiler` custom-op intercept) and Option 3 (upstream Nx
extension) remain available if the two-kernel-vs-fused gap materially
hurts a real workload after M11.

**Scope reductions from original PLAN.md M10.5**:

- **AWQ safetensors loader deferred.** The original plan called for a
  test-only loader that reads `Qwen/Qwen3-0.6B-AWQ` and maps to
  `%QuantizedWeight{}`. On closer inspection the AWQ→MLX conversion is
  meaningfully more involved than first thought: AWQ groups along the
  `in` axis while MLX's `transpose=false` path expects groups along the
  stored last axis, so correct conversion requires transposing the
  packed tensor, unpacking `qzeros` into per-group zero-points,
  computing `biases = -scales * zero_points`, and mapping HF param
  names to Bumblebee's internal naming. All tractable but substantial.
  The from-dense path above exercises the same defn-integration
  pipeline (graph rewrite + QW params + defn-native dequantize + JIT'd
  forward) and produces a useful regression gate; AWQ-specific loading
  is now a proper follow-up milestone rather than M10.5-scope.
- **Conformance oracle adjusted.** PLAN.md originally envisioned
  asserting against MLX Python's output on the same quantized
  weights. Since we're now quantizing Qwen3-0.6B ourselves (not
  loading AWQ), the reference is what this pipeline produces on a
  clean checkout — same discipline as `qwen3_full_test.exs`.

**Follow-ups** (out of M10.5 scope):

- AWQ safetensors loader + Bumblebee param-name mapping. When it
  lands, adds a second conformance test that loads real
  `Qwen/Qwen3-0.6B-AWQ` weights end-to-end.
- Optional upstream contribution to `deps/bumblebee` for AWQ loading.

### M11 — `mlx::fast::*` fused kernels

Orthogonal to the M6 generic-fusion drop. MLX ships handwritten fused
kernels that *do* beat composed defn on transformer hot paths:

- `mx::fast::scaled_dot_product_attention` — the QK^T → mask → softmax
  → V chain as one kernel with attention-mask broadcast handled
  internally. Replaces ~5 separate Native dispatches per attention
  layer per token.
- `mx::fast::rms_norm` — fused RMSNorm with epsilon and gain, replaces
  the `square → mean → rsqrt → multiply → multiply` chain.
- `mx::fast::layer_norm` — same story for LayerNorm (DistilBERT, ViT,
  Whisper encoder).
- `mx::fast::rope` — fused rotary embeddings with `traditional` /
  `default` mode flags, replaces the trig + reshape + interleave chain.

**Detection problem**: pattern-matching subgraphs inside `Nx.Defn.Expr`
to recognize "this is RMSNorm, emit one fused call" is compiler-level
work. Simpler v1 path: expose them as `Emily.Fast.*` Elixir helpers
callable from inside `defn`, and ship a thin Bumblebee shim that uses
them when the active backend is Emily. The pattern-matched compiler
version is a future follow-on.

**Testing**:
- Native unit tests against MLX's own test vectors.
- Backend equivalence: each `Emily.Fast.*` helper matches the
  composed-defn equivalent within a documented tolerance band (the
  fused kernels reorder ops slightly, so bit-match isn't expected).
- Re-run all four Bumblebee `:*_full` conformance suites with the
  fused-kernel path active; assert the existing logits slices still
  match.
- Pin a tokens/sec floor in `bench/qwen3_tokens_per_sec.exs`.

**Exit:** Qwen3, ViT, Whisper, DistilBERT all run with fused kernels
enabled and pass conformance; benchmark shows a measurable speedup
over the M9 baseline (target ≥1.5× on M3 hardware).

### M12 — Zero-copy binary round-trip (`to_binary`)

PLAN design decision #9 claims unified-memory zero-copy for
`from_binary` / `to_binary`. The current code memcpys unconditionally
(`emily_nif.cpp:57-58`, `:74-84`). M12 delivers the claim for
`to_binary`. `from_binary` retains its memcpy — the one-time cost at
model load is negligible relative to the file I/O that precedes it,
and Metal's `newBufferWithBytesNoCopy` requires page-aligned,
page-sized memory that real-world binaries (Bumblebee/safetensors)
never provide. See dropped M12.5 below.

- **`to_binary`**: wrap the materialized MLX buffer pointer as a BEAM
  resource binary via `enif_make_resource_binary`, with the resource
  retaining a refcount on the MLX array so the buffer survives until
  the BEAM binary is GC'd. No copy; the BEAM binary aliases MLX
  storage directly.
- **`from_binary`**: memcpy retained. Acceptable cost — see M12.5.
- **Stride-aware materialize**: `to_binary` currently routes through
  `mx::contiguous`; for already-contiguous arrays this is a no-op, but
  the wrap-as-resource path needs an explicit guard since aliasing a
  non-contiguous buffer would lie about its layout.

**Testing**:
- Allocate a tensor, call `to_binary`, assert MLX active memory did
  not grow (aliasing, not copying).
- Soak: repeated `to_binary` with cache-clear, assert peak memory is
  bounded by the working-set size.
- Correctness: the M2 property suite must still pass — this is a perf
  change, not a semantics change.
- Refcount safety: drop the original tensor reference, then read the
  BEAM binary returned by `to_binary`; must not segfault. Use-after-
  free is the failure mode, so this milestone gates on an
  AddressSanitizer build in CI.

**Exit:** `to_binary` zero-copy verified by allocator stats; M2
property suite green; lifecycle and soak tests verify refcount
safety. AddressSanitizer CI deferred (macOS SIP prevents
`DYLD_INSERT_LIBRARIES` propagation through `/bin/sh`-launched BEAM;
requires a custom `--enable-sanitizers=address` OTP build).
`EMILY_ASAN=1` Makefile flag ships for users with sanitizer-enabled
OTP.

### M12.5 — `from_binary` zero-copy via MTL no-copy buffer *(dropped)*

Dropped. The approach required `MTL::Device::newBufferWithBytesNoCopy`
which needs page-aligned (4096 B), page-sized memory. Real-world
binaries from Bumblebee/safetensors never meet these preconditions:
`:file.pread` allocates at 8–16 byte alignment, tensor sizes are
determined by model dimensions (not page multiples), and the
safetensors format packs tensors contiguously with no inter-tensor
padding. ~99 %+ of calls would fall back to memcpy regardless.

Additional complexity — MLX's private residency API, Metal framework
linking, persistent `ErlNifEnv` lifecycle — was disproportionate to
the benefit, which is a one-time cost at model load (not in the
inference hot path).

### M13 — EXLA gradient conformance

M9's grad oracles are `Nx.BinaryBackend` symbolic grad and f32 finite
differences. Both can be wrong in the same direction — BinaryBackend's
symbolic grad is the same `Nx.Defn.grad` lowering Emily uses, so a bug
in the lowering passes both oracles. Finite differences have their own
ulp floor and edge-case blind spots (NaN/inf/denormal, near-zero
saddles).

EXLA-on-Linux+CUDA is the missing third opinion. M3 already
established the CI pattern: run a model on EXLA, check in golden
outputs, assert Emily matches. M13 extends it to gradients.

- **CI job**: a Linux+CUDA runner produces a JSON file of
  `{function_id, input_seed, expected_grad_bytes}` for each function
  in M9's grad zoo. Checked into the repo; refreshed when the zoo
  changes.
- **Test harness**: `test/emily/grad/exla_oracle_test.exs`,
  `@moduletag :grad_conformance`. Loads the JSON, runs the same
  function under `Emily.Compiler` with the same seed, asserts grad
  matches the EXLA-produced bytes within a tolerance tighter than
  BinaryBackend's (EXLA uses the same hardware-vendor kernels Emily
  aspires to — the gap should be small).
- **Coverage**: M9 zoo plus a small transformer-block training step
  (forward + grad + SGD update of all parameters) so we catch per-op
  grad bugs that only manifest under composition.

**Testing**: the harness *is* the test. Tolerance calibration: pilot
3–4 ops first, document the Emily-vs-EXLA gap per dtype, ship per-op
tolerance tables — not a global epsilon.

**Exit:** `:grad_conformance` green on default CI; tolerance tables
checked in alongside the goldens.

**Shipped**:

- **Scope change**: EXLA CPU backend on macOS instead of Linux+CUDA.
  XLA-CPU is still a fully independent oracle (different compiler,
  different kernels from both BinaryBackend and MLX). CUDA conformance
  deferred to post-1.0.
- **`Emily.GradZoo`** (`test/support/grad_zoo.ex`) — extracted the 8
  `defn` grad functions and `softmax_last/1` from
  `grad_equivalence_test.exs` into a shared module. Added `fixed_inputs/1`
  (deterministic BinaryBackend tensors per function) and
  `grad_function/1` (function captures). Both existing grad test files
  updated to import from GradZoo.
- **`Emily.ExlaGoldenData`** (`test/support/exla_golden_data.ex`) —
  EXLA 0.11.0 CPU-generated golden gradient values for all 8 zoo
  functions plus a 1-step transformer-block training step (forward +
  grad + SGD update of all 8 parameters). Inline Elixir float lists,
  consistent with the existing conformance golden pattern.
- **`Emily.Grad.ExlaOracleTest`** (`test/emily/grad/exla_oracle_test.exs`)
  — `@moduletag :grad_conformance`. Per-function tolerance table
  (tighter than BinaryBackend's 1e-3: linear ops at 1e-6/1e-5,
  compositions at 1e-4/1e-3). Runs in the default test suite.
  `grad_dropout` excluded (PRNG divergence across backends).
- **Golden generator** (`bench/exla_golden_gen.exs`) — standalone
  Elixir script using `Mix.install` for `{:exla, "~> 0.10"}`. Runs on
  macOS (CPU) or Linux+CUDA. Emits a complete `ExlaGoldenData` module:
  `elixir bench/exla_golden_gen.exs`.

### M14 — Serving concurrency: stream-per-process + cookbook

`Emily.Compiler.__partitions_options__/1` raises on
`max_concurrency > 1` — correct (Metal isn't safe for concurrent
kernel dispatch from multiple OS threads), but it silently means a
single Emily-backed `Nx.Serving` cannot scale past one concurrent
request. Production users will hit this in week one. M14 stops being
silent about it and ships a tested pattern.

- **Stream-per-process**: the primary deliverable. Expose
  `Native.set_default_stream/1` and `Emily.Stream.with_stream/2`
  via MLX's `mx::scheduler::new_stream`. Each process gets its own
  Metal command queue; one shared model, per-process streams, no
  weight duplication. (Promotes streams from internal-only — see
  Project Decisions — to a narrowly-scoped public surface.)
- **Cookbook: pooled servings**: documented pattern — "for K
  concurrent inference requests, start K `Nx.Serving` instances
  behind your own pool (poolboy, Registry round-robin, etc.)".
  No library code; clients bring their own pool since Emily already
  behaves correctly under that model. Trade-off: each pool member
  loads its own weights — fine for small models, painful for
  Qwen3-7B+.
- **README + moduledoc updates**: surface the limitation and both
  patterns prominently. Today neither is mentioned outside a buried
  comment in `Emily.Compiler`.

**Testing**:
- Stream test: two processes, two streams, same model loaded once;
  no SIGSEGV under sustained parallel load
  (`test/soak/backend_concurrency_test.exs` documents the SIGSEGV
  story for the unstreamed case — this is the negative control).
- `:serving_full` opt-in: end-to-end per-stream large-model
  pattern.

**Exit:** both patterns documented; concurrency soak demonstrates
the streamed path is stable.

**Post-M14 note — eval serialisation:**
MLX is not thread-safe (ml-explore/mlx#2133). The Metal
`CommandEncoder` is shared state — concurrent `mx::eval` calls from
different OS threads crash with `"A command encoder is already
encoding to this command buffer"` (SIGABRT) or SIGSEGV from corrupted
encoder state. The M14 soak tests (8 concurrent workers via
`Task.async_stream`) exposed this by being the first code path to call
`mx::eval` from multiple dirty-CPU scheduler threads simultaneously.

Fixed by:
1. **`emily::safe_eval()`**: a mutex-serialised `mx::eval` wrapper in
   `c_src/emily/tensor.hpp`. All eval callsites route through it.
   Graph-building ops (regular scheduler) remain lock-free.
2. **Removed `set_default_stream` from `with_stream/2`**: the NIF
   mutated MLX thread-local state on BEAM scheduler threads, which
   is unreliable since BEAM processes migrate between OS threads.
   The process-dictionary-based stream routing was already correct.
3. **Hardened `resolve_stream(-1)`**: the -1 fallback now reads the
   device default directly instead of the (potentially corrupted)
   thread-local default.

The mutex serialises Metal dispatch at the cost of true concurrent
GPU execution. See M14.5 (MLX upgrade) for the plan to restore
concurrency via MLX's native thread-local `CommandEncoder` support.

### M14.5 — MLX upgrade (build from source)

Emily pins MLX 0.25.1 via pre-built binaries from `cocoa-xu/mlx-build`.
MLX gained native thread-safety on `main` in April 2026 (thread-local
`CommandEncoder` ml-explore/mlx#3348, `ThreadLocalStream` C++ API
ml-explore/mlx#3405), but neither fix is in any release yet (latest:
0.31.1). Building from source unblocks true concurrent Metal dispatch
and removes the `safe_eval` mutex introduced in the post-M14 fix.

- **Build MLX from source** (from `main` or 0.32+ when released)
  instead of fetching the pre-built 0.25.1 tarball. Extend `mix.exs`
  to support an `MLX_SOURCE` env var pointing at a local MLX build.
- **Audit API changes** between 0.25.1 and target version. Emily's
  C++ surface is narrow (core ops, linalg, streams, eval, allocator),
  but six months of MLX releases may rename or remove functions.
- **Adopt `ThreadLocalStream` C++ API**: each BEAM dirty-CPU thread
  gets its own MLX stream automatically, enabling true per-thread
  Metal command queues without the eval mutex.
- **Remove `emily::safe_eval` mutex** once native thread-safety is
  validated — revert to direct `mx::eval` calls.

**Testing**:
- Amplified stress test (16 workers, 100 iterations) passes without
  mutex under the new MLX build.
- Full test suite 20x with zero crashes.
- Benchmark `to_binary` latency under concurrent load: confirm
  throughput improves vs. the mutex-serialised path.

**Exit:** concurrent soak tests pass without mutex; MLX build-from-source
documented; stress test confirms concurrent Metal dispatch is stable.

### M15 — Native linalg

`lu`, `svd`, `qr`, `cholesky`, `triangular_solve`, `eigh`,
`determinant`, and friends route through `via_binary` today. Correct,
BinaryBackend-slow. MLX exposes most natively under `mx::linalg::*`.

- Bind each available `mx::linalg::*` function as a Native NIF.
- Replace the `via_binary` Backend callbacks with Native dispatch.
- Document divergences (MLX's pivot strategy may differ from Nx's
  reference; numerical conditioning thresholds may differ).

**Testing**:
- Native unit tests against hand-computed references for small
  matrices (3×3, 4×4) where the answer is checkable.
- Backend property tests vs. `Nx.BinaryBackend` with shape generators
  biased toward well-conditioned inputs (random Gaussian → QR →
  reconstruct). Document the conditioning-bound failure mode for
  ill-conditioned cases.
- Existing `via_binary` fallbacks for any op MLX doesn't implement
  natively; add a fallback-coverage test for the residual.

**Exit:** all `mx::linalg::*`-backed callbacks pass property suite;
remaining `via_binary` linalg paths documented with rationale.

**Post-M15 note — intermittent test crashes:**
Two crash modes were observed during M15 development.

1. **SIGABRT (exit 134) — fixed:** LAPACK errors (SVD convergence, LU
   singular matrix) abort the VM because MLX's `StreamThread::thread_fn`
   has no catch frame — any C++ exception from an `eval_cpu` primitive
   hits `std::terminate`. This is an MLX design constraint, not
   something the NIF layer can catch. Fixed by strengthening test
   inputs:
   - SVD property test now applies `make_well_conditioned/1` (was the
     only linalg test without it).
   - `make_well_conditioned/1` multiplier raised from `n*10` to
     `n*10+20` — the old value landed on the diagonal-dominance
     boundary for n=2, allowing f32 rounding to produce singular
     pivots in LAPACK.
2. **SIGSEGV (exit 139) — pre-existing, not M15-related:** Reproduces
   at ~4/10 on main with no linalg tests. Likely an MLX Metal driver
   issue (see `test_helper.exs` commentary). The linalg NIFs add
   `mx::eval()` on inputs before the cross-stream `cpu_stream()`
   handoff as a defensive measure, but this does not fix the underlying
   SIGSEGV. Needs separate investigation outside the M15 scope.

### M16 — Mixed-precision training

bf16 activations + f32 master weights + loss scaling is the standard
recipe for Qwen-scale training. Emily accepts bf16/f16 at the backend
but ships no policy, no tolerance tables, no loss-scale primitive.
Promotes "mixed-precision master weights" out of v1 non-goals.

- **`Emily.MixedPrecision`**: thin Elixir module exposing
  `cast_params/2` (downcast f32 → bf16 for forward),
  `accumulate_grad/2` (upcast bf16 grad → f32 for the optimizer step),
  `loss_scale/1` / `unscale/2` (dynamic loss scaling with overflow
  detection on `isfinite` reductions).
- **Backend policy**: bf16 ops dispatch to MLX bf16 kernels (already
  supported); f32 master weights live alongside. No type-promotion
  surprises — the user explicitly casts at the forward/backward
  boundary.
- **Tolerance tables**: per-op, per-dtype tolerances per the M9
  harness. bf16 has ~3 decimal digits of precision; property tests
  must use bf16-appropriate epsilons, not f32 ones.

**Testing**:
- Grad equivalence under bf16 with f32 accumulation: extend M9's zoo
  with bf16 cases, assert match within bf16 tolerance.
- Loss scaling: deliberately overflow at bf16, assert the
  loss-scale dynamic adjustment (halve scale on overflow, double every
  N successful steps) reaches a stable scale.
- Convergence canary: extend `:training_full` MNIST with a bf16
  variant; assert >97% accuracy still reached.

**Exit:** bf16 grad equivalence green; MNIST bf16 convergence within
0.5% of the f32 baseline; loss-scaling primitives documented with a
worked example in the moduledoc.

**Shipped.**

- `Emily.MixedPrecision` (`lib/emily/mixed_precision.ex`) — `cast_params/2`,
  `accumulate_grad/2`, `loss_scale/1`, `scale_loss/2`, `unscale/2`,
  `update/2`, `has_overflow?/1`. Nested `LossScaler` struct with dynamic
  scaling (halve on overflow, double every N successful steps, floor at
  `min_scale`). Moduledoc includes a complete mixed-precision training
  step as a worked example. Traversal covers maps, tuples, lists, and
  `%Nx.Tensor{}` leaves; `Nx.Container` structs (e.g. `Axon.ModelState`)
  are not traversed — documented.
- bf16 grad equivalence (`test/emily/grad/bf16_grad_equivalence_test.exs`)
  — all 8 zoo functions pass under Emily.Compiler vs BinaryBackend
  Evaluator, both in bf16, within 1e-2 tolerance.
- Mixed-precision MLP curve-matching
  (`test/emily/training/bf16_mlp_curve_test.exs`) — 50-step training
  loop with f32 master weights, bf16 forward pass, loss scaling, and f32
  gradient accumulation. Emily vs BinaryBackend within rtol 5e-2.
- bf16 MNIST convergence canary
  (`test/emily/training/mnist_bf16_full_test.exs`) — `:training_full`,
  uses `Axon.MixedPrecision.create_policy`. Target ≥ 95.5%.
- Backend `coerce` fix: `Emily.Backend.wrap` now checks `Native.dtype(ref)`
  and casts if the MLX buffer dtype disagrees with the declared Nx output
  type. Previously only handled pred→u8; now handles all type mismatches
  (e.g. bf16 buffer with f32 metadata from `Nx.Defn.grad` type promotion).

### M17 — Conv-pool training (was M10)

Originally scoped as M10 in the pre-review plan. Re-prioritized below
the inference, oracle, serving, linalg, and mixed-precision
milestones because its reach is narrower — small-CNN training is a
real but limited use case relative to "make Bumblebee inference
production-ready".

Lift window reductions (`window_sum`, `window_max`, `window_min`,
`window_product`, `window_scatter_max`, `window_scatter_min`) off
`via_binary` onto their native MLX counterparts. Closes the last gap
in the training primitive set and unblocks pool-based conv models
(small CNNs, ViT classifier heads trained from scratch).

Scope is unchanged: the lifts are mechanical per-op changes. Test
coverage extends the M9 grad-equivalence and curve-matching zoo to
cover the new ops, plus a small-CNN MNIST run in `:training_full`.

**Exit:** grad-equivalence on window ops green; small-CNN MNIST
training converges in `:training_full`.

**Shipped.**

- `c_src/ops/pooling.cpp` — six new NIFs. Reductions composed as
  `mx::pad` → `mx::as_strided` (sliding-window view) → reduce, mirroring
  `vendor/mlx/python/mlx/nn/layers/pooling.py` generalised to N-D.
  Scatter variants add argmax-with-last-occurrence-tie-break
  (`mask * arange(K)` + argmax, since MLX's native argmax is
  first-occurrence) → per-axis absolute-index computation →
  `mx::scatter_add` into a `mx::full(init_value)` padded buffer →
  slice back to the input shape.
- `Emily.Native`: 6 new NIF stubs
  (`window_sum`/`max`/`min`/`product`/`scatter_max`/`scatter_min`).
- `Emily.Backend` (`backend.ex:1101-1225`): `apply_window_reduce/5` and
  `apply_window_scatter/6` helpers; `:valid`/`:same` padding resolution
  and dtype-specific identity (0/1/±∞, plus `{:s, _}` / `{:u, _}`
  min/max) done in Elixir before the NIF.
- Forward-parity coverage:
  `test/emily/backend_window_test.exs` (19 unit tests across shape ×
  stride × padding × dilation × f32/bf16/s32/u8),
  `test/emily/backend_window_scatter_test.exs` (11 tests including
  overlapping windows, tie-break, non-zero `init_value`, 1-D/3-D).
- Grad-equivalence extensions (`test/support/grad_zoo.ex` +
  `test/emily/grad/grad_equivalence_test.exs`): three new zoo fns —
  `grad_window_sum`, `grad_window_max_pool` (lands on
  `window_scatter_max` via Nx's grad rule), and `grad_window_avg_pool`.
  `bf16_grad_equivalence_test.exs` auto-picks-up the new zoo entries.
  The EXLA oracle skips un-regenerated zoo entries via
  `ExlaGoldenData.has_golden?/1` — run `mix run
  bench/exla_golden_gen.exs` to add window goldens in a follow-up.
- `test/emily/training/cnn_curve_test.exs` — handwritten 2-conv +
  max-pool CNN, 30-step SGD, per-step loss trajectory match vs
  BinaryBackend within rtol 1e-2 (looser than the MLP curve test
  because the CNN stacks four reductions).
- `test/emily/training/mnist_cnn_full_test.exs` — `:training_full`
  LeNet-style Axon CNN on MNIST, ≥ 97% test accuracy. Validated:
  5-epoch loss 1.60 → 0.12, test accuracy ≥ 97% on a 64-batch run.

### M18 — Observability & fallback telemetry

Hitting `via_binary` is ~100× slower than native and emits no signal.
Whisper before M8 spent 90% of forward-pass time in a BinaryBackend
round-trip with no log. The same shape of bug will keep happening as
ops rotate on/off `via_binary` — make it observable.

- **`:telemetry` events** at each Native dispatch, fallback entry,
  and evaluation boundary. Span-style start/stop so consumers can
  build histograms.
- **Fallback warning**: configurable via app env
  (`config :emily, warn_on_fallback: true`), emits a one-shot
  `Logger.warning` per `{op, input_shape}` so a Bumblebee user sees
  "indexed_put on shape X fell back to BinaryBackend" once, not every
  forward pass.
- **Allocator/peak-memory telemetry** wired so a long-running serving
  can graph memory drift without manual `get_active_memory` polling.

**Testing**:
- Attach a test handler, run a known fallback op, assert the event
  fires.
- Assert one-shot dedup of the fallback warning over 100 calls.

**Exit:** events documented in `Emily.Telemetry` moduledoc; fallback
warning behavior covered by tests.

**Shipped.**

- `Emily.Telemetry` (`lib/emily/telemetry.ex`) — moduledoc enumerates
  every event; `memory_stats/0` samples the MLX allocator and emits
  `[:emily, :memory, :stats]`. `maybe_warn_fallback/2` and
  `init_dedup_table/0` are the internal helpers; dedup state lives in
  a named `:public` ETS table owned by `Emily.Application`.
- Span events via `:telemetry.span/3`: `[:emily, :eval, *]` on
  `Emily.eval/1`, `[:emily, :to_binary, *]` on both `Emily.to_binary/1`
  and `Emily.Backend.to_binary/2` (the Nx.to_binary path) with
  `:shape`/`:dtype`/`:byte_size` metadata, and
  `[:emily, :fallback, *]` on every `via_binary` entry with
  `:op`/`:input_shapes`/`:input_dtypes`.
- One-shot `Logger.warning` per `{op, input_shapes}` pair. Opt-in via
  `config :emily, :warn_on_fallback, true`; off by default so library
  consumers and CI logs stay quiet. The telemetry event fires
  regardless — the log is a dev-time convenience on top of it.
- **Scope interpretation.** PLAN's "each Native dispatch" was read as
  the evaluation boundary (`Native.eval` / `Native.to_binary`) rather
  than wrapping 300+ graph-construction call sites in
  `Emily.Backend`. Graph-construction NIFs are <10μs and do no work;
  the evaluation boundary is where MLX actually runs kernels, and
  it's the point every lazy tensor funnels through. If per-op
  graph-construction histograms are ever needed, the right answer is
  a centralised dispatch helper, not per-callback decoration.
- **Op-name plumbing.** `via_binary/3` → `via_binary/4`;
  `via_binary_tuple/3` → `via_binary_tuple/4`; `apply_scatter/7` →
  `/8`. The op name propagates into both the `:telemetry` metadata
  and the dedup key.
- Tests: `test/emily/telemetry_test.exs` covers fallback start/stop
  events, 100-call dedup via `capture_log`, `warn_on_fallback=false`
  silence, `to_binary` span metadata, and `memory_stats/0` emission.
  `async: false` because the dedup table is global.

### M19 — Error surfacing *(deferred — post-1.0)*

*Deferred — post-1.0.* The original scope — typed exception hierarchy
(`Emily.ShapeError`, `Emily.DtypeError`, `Emily.MLXError`) with
per-NIF wrapping to carry op name + input shapes/dtypes — remains
sound, but three factors argue for waiting:

1. **Nx validates upfront.** Shape/dtype mismatches raise in
   `Nx.Shape.*` / `Nx.Type.*` before dispatch hits `Emily.Backend`,
   with structured messages. MLX's raw surface only leaks through for
   direct `Emily.Native.*` callers (tests, `Emily.Quantization`) and
   numerical failures Nx can't validate (singular matrix, SVD
   non-convergence) — a narrow blast radius in practice.
2. **Ecosystem parity.** No Nx backend (EXLA, BinaryBackend, EMLX,
   Torchx) ships a typed exception hierarchy. Emily is not worse than
   peers today; raw `ArgumentError` / `RuntimeError` with the vendor
   message verbatim is the ecosystem convention.
3. **Hot-path cost.** Per-NIF try/rescue wrapping adds ~70–100 ns per
   call (extra function call + try frame). At transformer-inference
   NIF counts (~10k calls per Qwen3 forward pass), that's ~1 ms of
   trace-time tax for errors that mostly don't happen.

When we revisit — likely bundled with a 1.x → 2.0 bump, since
introducing rescue-able exception types is a contract change users
may come to depend on — the plan is the one originally written for
M19: three `defexception` modules carrying `:op`, `:input_shapes`,
`:input_dtypes`, `:message`, `:callback`; macro-generated wrappers
over a renamed `Emily.Native.Nif`; a `wrap_callback/2` helper in
`Emily.Backend`; migration of the ~13 existing `ArgumentError` raises
in `lib/emily/backend.ex`, `lib/emily/quantized_weight.ex`,
`lib/emily/quantization.ex` (keep API-misuse raises — device,
`max_concurrency`, unknown options — as `ArgumentError` per Nx
convention).

**Exit (deferred):** revisit at 2.0 planning.

### M20 — GPU interop pointers *(deferred — post-1.0)*

*Deferred — post-1.0.* The original scope — implementing
`Nx.Backend.from_pointer/5` and `to_pointer/2` over MLX buffer pointers
with caller-supplied deleters — remains reachable but is not the right
shape for Emily's integration surface:

1. **Elixir can't represent a raw pointer.** `Nx.Pointer.address` is a
   uint; a user can't dereference it. The only thing they can do with a
   `to_pointer` result is hand it to another NIF. Elixir is a passive
   integer shuttle, not the real API.
2. **Ecosystem parity.** Neither EMLX nor Torchx exposes pointer ops in
   Elixir; `Nx.BinaryBackend` raises too. Emily is not worse than peers
   today — raw `raise` with a clear message is the ecosystem
   convention for backends that haven't implemented the callbacks.
3. **MLX DLPack lives in the Python bindings**
   (`vendor/mlx/python/src/convert.cpp`), not the C++ public API. No
   clean capsule path to lift.
4. **No concrete consumer.** The correct design is a public C++ header
   (`include/emily.h`) exporting the Tensor resource type and opaque
   `mlx::array` accessors so downstream NIFs can interop in C++ — a
   library-packaging task needing ABI stability commitments. Designing
   that speculatively risks shipping the wrong shape.

When we revisit — prompted by a real user with a native pipeline or a
DLPack-importing tool — the plan is the one originally written for
M20, restructured as a NIF-to-NIF contract rather than an Elixir
surface:

- Public header `include/emily.h` exporting:
  - `emily_tensor_resource_type(ErlNifEnv*)` — the `ErlNifResourceType*`
    so downstream NIFs can `enif_get_resource` Emily tensors.
  - `emily_array_from_term(ErlNifEnv*, ERL_NIF_TERM) -> mlx::array&` —
    read access to the underlying array.
  - `emily_array_to_term(ErlNifEnv*, mlx::array) -> ERL_NIF_TERM` —
    wrap a caller-constructed `mlx::array` (including one built from a
    user-owned pointer via MLX's `array::array(void*, shape, dtype,
    Deleter)` constructor at `vendor/mlx/mlx/array.h:69`) as a Tensor
    resource.
- `Nx.Backend.from_pointer/5` and `to_pointer/2` continue to raise,
  with messages pointing at the header-based interop path.
- Lifetime contract documented explicitly: the `Deleter` closure owns
  the backing buffer; the Tensor resource refcount gates the
  `mlx::array` lifetime; downstream NIFs must hold the resource, not
  the raw pointer.
- Testing: a minimal downstream NIF harness in `test/c/` exercising
  allocate-in-emily → borrow-pointer-in-foreign-NIF → release, under
  AddressSanitizer (subject to the same macOS-SIP caveat recorded in
  M12's exit notes).

**Exit (deferred):** revisit when a concrete downstream consumer asks
for pointer interop.

### M21 — `mix emily.doctor` *(deferred — post-1.0)*

*Deferred — post-1.0.* The original scope was a diagnostic Mix task
covering the MLX prebuilt fetch + checksum + `elixir_make` chain —
the failure surface on fresh macOS machines when Emily pinned the
`cocoa-xu/mlx-build` prebuilt. M14.5 moved Emily to building MLX from
source, which collapses most of that failure surface:

1. **No prebuilt fetch to probe.** Checksum mismatch, cache corruption,
   and registry-unreachable paths no longer exist. The remaining build
   failures (missing Xcode CLT, CMake version skew, out-of-disk)
   surface as `elixir_make` errors during `mix compile` with messages
   users can already read.
2. **NIF-load and GPU-dispatch probes duplicate the test suite.** A
   successful `mix test` already exercises both paths more thoroughly
   than a single probe would. Users who can't run tests can't run
   `mix emily.doctor` either.
3. **No concrete adoption signal yet.** The task was motivated by
   speculative day-one friction; we haven't seen the issues it
   targets filed against the repo.

When we revisit — prompted by a real pattern of setup failures that
`elixir_make` errors don't already explain — the plan is a narrower
version of what was originally written:

- Probe Xcode CLT version, CMake presence/version, MLX source-tree
  checkout state, compiled NIF load, trivial GPU dispatch, and
  available unified memory.
- Drop the prebuilt-presence and prebuilt-checksum probes from the
  original scope.
- Structured report with green/yellow/red per probe and remediation
  hints keyed to the observed failure modes.
- Snapshot test on a known-good configuration; negative tests that
  deliberately break one probe at a time.

**Exit (deferred):** revisit if adoption-friction issues accumulate
around the source-build path.

### M22 — Compile-time debug flags

Tracks [issue #32](https://github.com/ausimian/emily/issues/32). During
DistilBERT-QA conformance testing we hit an intermittent `:nan` score
when a vocab-30522 tokenizer was paired with a tiny-random model whose
embedding table had only 1124 rows. The tokenizer emitted out-of-range
token ids, and `Emily.Backend.gather/4` silently returned uninitialised
memory for the OOB rows — sometimes benign floats, sometimes NaN that
propagated through softmax. `Nx.BinaryBackend` raises on OOB gather
because the check is free on CPU; GPU backends (Emily, EXLA,
Torch-CUDA, JAX-GPU) dispatch to kernels that do not bounds-check.
That's the GPU-backend norm, not an Emily bug — but it hides a real
class of caller error.

M22 lands a compile-time opt-in debug-assertion facility so users can
re-enable these checks in CI / test / dev builds at zero default cost.

**Mechanism.** `Application.compile_env(:emily, :<flag>, false)` module
attribute + dead `if` branches. When the flag is `false` (default), the
Elixir compiler eliminates the assertion branch — zero runtime cost,
zero bytecode. Mix recompiles affected modules automatically when the
flag flips. Verified by inspecting `.beam` disassembly for the
default-off case, and by relying on `Application.compile_env/2`'s
compile-vs-runtime divergence warning for the flip case.

**Initial flag set** (coherent subset, not every flag from the issue):

- `:debug_bounds_check` — assert indices in range for indexing ops:
  `gather` / `take` on reads and `scatter` / `scatter_add` on writes.
  One flag covers both because the failure class is identical (OOB
  indexing into a resource tensor) and users almost always want both
  — in inference `gather` is the hot path, in training `scatter`
  fires through the grad of `gather`, and the DistilBERT-QA bug
  class shows up via either. Can be split into per-op flags if a
  real user later asks for cost-profile granularity.
- `:debug_detect_nan_inf` — post-op NaN/Inf scan on the hot training
  ops (matmul / softmax / layer-norm), so training-time numerics
  failures surface at the op that produced them rather than as a
  downstream `loss = NaN`.

Deferred to follow-ups if demand materialises:

- `:debug_assert_same_backend` — detect tensors that silently moved
  between `Emily.Backend` and `Nx.BinaryBackend` mid-graph.
- `:debug_strict_dtype` — raise on silent mixed-dtype promotion.

**Naming and default.** `:debug_*` prefix across the set. Every flag
defaults to `false` — explicit opt-in only, including in Emily's own
`config/test.exs`. The per-flag unit tests (below) exercise the
flag→assertion path; there's no value in Emily's CI paying GPU-sync
cost on every `mix test` run, and running Emily's own tests in the
same configuration users hit in production avoids masking perf
regressions. Each flag documented in the `Emily` moduledoc under a
"Debug assertions" heading with a worked `config/test.exs` snippet
consumers can lift into their own projects.

**Testing.**

- Per-flag unit tests: flag off, OOB op succeeds silently (negative
  control); flag on, OOB op raises with a message identifying op +
  offending index / value. `:debug_bounds_check` gets coverage on
  both a gather-family op (`gather` / `take`) and a scatter-family
  op (`scatter` / `scatter_add`) since one flag now covers both.
- Zero-cost verification: a single test that compiles with all flags
  off and asserts the assertion-branch `fn` does not appear in the
  module's `beam_disasm` output. Guards against accidental runtime
  cost from a future refactor.
- `:debug_detect_nan_inf` gets a training-loop test: inject a NaN
  (divide-by-zero or overflow) into a known op, assert the flag
  surfaces the failure at that op's boundary rather than at the
  final loss.

**Risks.**

- Each assertion is a GPU sync (breaks lazy-graph fusion). Documented
  loudly alongside each flag; the default-off story is the mitigation.
- `:debug_detect_nan_inf` is a reduction per op — meaningful overhead
  even off the hot path. Scope to a short allow-list of ops the
  issue named (matmul / softmax / layer-norm), not every op.

**Exit.** Two flags shipped with tests; all flags default-false,
including in Emily's own `config/test.exs`; README and `Emily`
moduledoc document the pattern with a worked opt-in snippet; zero-cost
verification test asserts the assertion branches compile out when the
flags are off.

### M23 — Public documentation & examples review

Pre-1.0 pass over the documentation surface users actually consume —
ExDoc-rendered module docs on HexDocs, the README, and worked
examples. Not a rewrite; a structured audit + gap-fill.

**Current state** (surveyed pre-milestone):

- All 9 public modules have moduledocs. Design rationale (Backend
  divergences, Compiler cache story, Quantization dispatch, Stream
  concurrency, MixedPrecision worked example) is explained where it
  matters.
- `Emily.Backend` exposes 83 public functions with only 6 `@doc`
  attributes — the 77 undocumented are `Nx.Backend` callback
  implementations users should not call directly.
- `Emily.Compiler` exposes 4 public functions with 0 `@doc` — same
  pattern, `Nx.Defn.Compiler` callbacks.
- No `.livemd` notebooks, no `examples/` directory. `bench/` is
  internal tooling. The README points onboarding readers at
  `test/emily/conformance/` for worked model usage, which is awkward.
- `mix.exs` `docs:` config is minimal: `main: "readme"`, three
  extras. No `groups_for_modules`, no HexDocs link in the README.
- `CHANGELOG.md` is a stub; `RELEASE.md` holds the working M13–M18
  notes. Cutover process not documented.

**Scope.**

- **Partition `Emily.Backend` / `Emily.Compiler` callback
  implementations from the public surface.** The 77 Backend + 4
  Compiler callback functions get `@doc false` (Nx backend convention —
  see `deps/nx/lib/nx/binary_backend.ex` for precedent). The six
  functions genuinely intended for direct use keep their `@doc`. A
  "Public API" section in each moduledoc enumerates what users call
  directly vs what Nx dispatches to on their behalf.
- **`mix.exs` `groups_for_modules`**: organise the nav by concern —
  Core (`Emily`, `Emily.Backend`, `Emily.Compiler`), Concurrency
  (`Emily.Stream`), Quantization (`Emily.Quantization`,
  `Emily.QuantizedWeight`), Training (`Emily.MixedPrecision`),
  Performance (`Emily.Fast`), Observability (`Emily.Telemetry`).
- **README: add HexDocs link** (`https://hexdocs.pm/emily`) under a
  "Documentation" heading. Link to the Livebook(s) below.
- **`notebooks/` directory with at least two `.livemd` files**:
  1. `notebooks/distilbert_qa.livemd` — smallest useful worked
     example; mirrors the M3 conformance test without the test
     harness. Telemetry-attach demonstrated inline.
  2. `notebooks/qwen3_quantized.livemd` — the headline use case;
     loads Qwen3-0.6B, quantizes via `Emily.Quantization.Transform`,
     greedy-decodes under `Bumblebee.Text.generation/4`. Shows
     `Emily.Stream` for concurrent serving.
  Both use `Mix.install/2` at the top so they run standalone without
  checking out the repo.
- **Pass over each public moduledoc** for 1.0-readiness: stale
  milestone references removed, examples run under the current API,
  deliberate divergences from Nx documented as a consistent
  "Divergences from `Nx.BinaryBackend`" subsection where applicable.
- **CHANGELOG.md cutover**: move the M13–M18 entries from
  `RELEASE.md` into `CHANGELOG.md` under a `0.1.0 (unreleased)`
  heading during this milestone; document the expected cutover in
  the release process so future milestones know which file to
  touch.

**Explicitly out of scope** (pattern reuse, not polish):

- A full guides directory (`guides/*.md`). The moduledocs carry the
  design material today; promoting them to separate guides is a 2.0
  concern once the surface stabilises.
- Contribution docs (`CONTRIBUTING.md`, issue/PR templates) beyond
  whatever the `/adopt-build-conventions` skill already ships.

**Testing.**

- `mix docs` runs clean (no warnings about missing modules, broken
  links, or unresolvable cross-refs).
- Each `.livemd` file in `notebooks/` is executed end-to-end as a
  smoke test (CI job or `mix test.notebooks` Mix task) against the
  real Bumblebee models used in `:*_full` conformance. Gated the
  same way as `:qwen3_full` — opt-in, not default CI.
- A README link-check asserts no broken HexDocs / GitHub links.

**Exit.** `mix docs` output reviewed module-by-module; two Livebooks
land in `notebooks/` and execute cleanly; README has a Documentation
section with working HexDocs + Livebook links; `CHANGELOG.md` holds
the 0.1.0 draft; callback-impl noise removed from the HexDocs nav.

### M24 — 1.0 release (was M11)

- API docs, HexDocs, README with worked Bumblebee + quantized-Qwen3
  examples
- Hex release (public), versioned per conventions (`@version` in mix.exs)
- `RELEASE.md` accumulated across feature branches

### M25 — microscaled quantization modes

Thread a `:mode` string through the quantize / dequantize /
quantized_matmul pipeline so `Emily.QuantizedWeight` covers MLX's
full `QuantizationMode` enum (see `vendor/mlx/mlx/primitives.h:155`),
not just the classical `"affine"` int4/int8 scheme.

- `c_src/ops/linalg.cpp` — `quantize_nif`, `dequantize_nif`,
  `quantized_matmul_nif` take a `std::string mode` arg. The biases
  arg on the latter two becomes `std::optional` since MLX's
  `fp_quantize` returns only `(wq, scales)` for microscaled modes.
- `lib/emily/native.ex` — stub arities bumped (`quantize/5`,
  `dequantize/7`, `quantized_matmul/9`).
- `lib/emily/quantized_weight.ex` — new `:mode` field enforced in
  `@enforce_keys` / `defstruct` / `@type t` / `Nx.Container :keep`.
  `from_dense/2` accepts `:mode` (default `"affine"`, else one of
  `"mxfp4"`, `"mxfp8"`, `"nvfp4"`) and validates mode-specific
  `{group_size, bits}` before dispatching. Microscaled modes carry a
  scalar-zero placeholder in `:biases` (the Native layer substitutes
  `nil` before the MLX call).
- `lib/emily/quantization.ex` — `quantized_matmul/2` forwards
  `qw.mode` and passes `nil` biases for microscaled modes.
  `dequantize_defn/1` raises a clear `ArgumentError` on non-affine
  modes, pointing callers at `to_dense/1` (the Native path).
- Tests — per-mode round-trip and `quantized_matmul` equivalence vs
  `Nx.dot(x, Nx.transpose(to_dense(qw)))` for each microscaled mode;
  error path for `dequantize_defn` on `"mxfp4"`.
- Metal smoke tests ran green for all four modes on Apple Silicon
  (see agent report in the milestone feature branch).

**Deferred (B4b)** — MLX's `to_fp8` / `from_fp8` ops (`vendor/mlx/
mlx/ops.h:1517-1521`) need an FP8 dtype that Nx doesn't yet model.
Revisit when M16 (mixed precision) surfaces a concrete FP8 user story,
or when upstream Nx gains an FP8 variant. Options considered:
Nx upstream, shadow-type wrapper, opaque-handle operator — none a
clear win at Emily's current maturity.

## Testing philosophy

| Layer | Oracle | Harness |
|---|---|---|
| Native | Hand-computed expected values | ExUnit unit tests |
| Backend | `Nx.BinaryBackend` on the same inputs | StreamData property tests + Nx conformance |
| Compiler | `Emily.Backend` in non-defn mode | Equivalence tests (same function, two modes) |
| Grad | `Nx.BinaryBackend` grad + finite differences (+ EXLA goldens from M13) | StreamData property tests + numerical oracle (+ checked-in EXLA-produced goldens) |
| Training | `Nx.BinaryBackend` loss trajectory | Curve-matching; MNIST convergence (`:training_full`, opt-in) |
| E2E | EXLA-produced golden outputs | Conformance tests with cached weights |

A bug can only be introduced in the layer where its test fails — no
cross-layer mystery bugs.

Additional harnesses:
- **Memory soak** (`test/soak/memory_test.exs`, `@tag :soak`): 10k
  iterations; MLX memory stats asserted to return to baseline.
- **Training memory soak** (`test/soak/training_test.exs`,
  `@tag :soak`, from M9): 1k training steps; baseline restored after
  `clear_cache/0`.
- **Concurrency** (`test/soak/concurrency_test.exs`, `@tag :soak`):
  parallel inference; determinism + no crashes.
- **Benchmarks** (`bench/`): Benchee scripts; results logged in
  `RELEASE.md` per version.
- **Conformance vs EXLA** (CI matrix, Mac for Emily + Linux+CUDA for
  EXLA oracle): same model, same input; runs on every PR touching
  Backend.
- **Convergence** (`:training_full`, from M9; opt-in CI job): Axon
  training loop on MNIST; catches numerical drift the curve-matching
  oracle can't see.

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
- **Streams**: internal in v1 except for the narrow `Emily.Stream`
  surface M14 introduces for the documented "big model, multi-process
  serving" pattern.
- **Training**: in scope from M9 (autodiff + small-scale loops);
  extended by M16 (mixed precision) and M17 (conv-pool). Out of scope:
  distributed training and a native optimizer library.
- **EMLX coordination**: none — quiet ship.

## Capability audit: MLX 0.31.1+69 (2026-04-22)

Earlier milestones were planned against the cocoa-xu prebuilt MLX
(the same binary elixir-nx/emlx depends on, v0.25.1-era at the time
M6 was measured). The tree now vendors `8e649be4`
(`v0.31.1-69-g8e649be4`) and builds from source via CMake. This audit
re-verifies the ratified decisions above and catalogues in-roadmap
MLX capabilities Emily isn't yet using. Scope limited to what is
plausibly in Emily's roadmap; see "Re-affirmed out-of-scope" at the
bottom for the explicit rejections.

### M6 re-measurement

M6's original drop verdict (≥1.20× gate not met) stands on 0.31.1+69.
Transformer-block GPU speedup is now 1.09–1.11× median (up from
1.04–1.07×); CPU still regresses. See the "MLX 0.31.1+69" section in
[`bench/compile_microbench.md`](bench/compile_microbench.md). No
change to M6 status.

### In-roadmap capability gaps

| # | Capability | MLX location | Emily status | Roadmap fit | Effort | Recommendation |
|---|---|---|---|---|---|---|
| B1 | `einsum` with automatic contraction-path optimisation | `vendor/mlx/mlx/einsum.h:14,18` | No references in `c_src/` or `lib/`. `Nx.einsum` decomposes through the Backend via reshape/transpose/tensordot, so contraction ordering is left on the table. | ViT (M7) and any transformer variant with non-trivial contractions. | NIF-only wrap + `Emily.Backend.einsum/3` override. Low. | **Candidate** — wrap when a model surfaces a measurable contraction-ordering loss; not urgent. |
| B2 | `fast::scaled_dot_product_attention` attention sinks | `vendor/mlx/mlx/fast.h:47-55` (the `sinks` param) | Emily passes `std::nullopt` at `c_src/ops/fast.cpp:106`; the `Emily.Fast.scaled_dot_product_attention*` helpers (`lib/emily/fast.ex`) don't expose sinks. | Long-context streaming decode for Qwen3 (M4) and Qwen3-VL (M24). StreamingLLM-style sink tokens are the main use case. | Extend the existing `fast::sdpa` NIF; add a `defn`-callable helper in `Emily.Fast`. Medium (plumbing through the optional-expression hook). | **Candidate** — defer until a model target demands long-context decode; no immediate blocker. |
| B3 | Sparse / MoE matmuls: `gather_qmm`, `gather_mm`, `block_masked_mm`, `segmented_mm` | `vendor/mlx/mlx/ops.h:1524, 1568, 1578, 1591` | Emily wraps only `quantized_matmul` (`c_src/ops/linalg.cpp`). No MoE-dispatch path. | Qwen3-MoE variants, any sparse-attention model. Outside current M4/M7 targets. | NIF wrap + Backend override + design the `Emily.Quantization` API extension. Medium. | **Deferred** — surfaces with the first MoE model target; not a v1 blocker. |
| B4a | Microscaled quantization modes (`Mxfp4`, `Mxfp8`, `Nvfp4`) | `vendor/mlx/mlx/primitives.h:155` (`QuantizationMode` enum) | **Landed in M25.** `Emily.QuantizedWeight` now carries a `:mode` field; `quantize`/`dequantize`/`quantized_matmul` NIFs accept the mode string; Metal smoke-tested for all four modes (affine max_diff ~3e-5, microscaled modes match MLX's own dequant-then-dot oracle within f32). | — | Shipped. | — |
| B4b | FP8 dtype (`to_fp8` / `from_fp8`) | `vendor/mlx/mlx/ops.h:1517-1521` | **Deferred.** Requires either an FP8 variant in `Nx.Type` (upstream coordination) or a shadow-type wrapper like `QuantizedWeight`. Neither is clearly right without a concrete FP8 user story. Revisit with M16 (mixed precision) or when Nx gains FP8. | M16 extends mixed precision to FP8 master weights. | Blocked on Nx dtype or shadow-type design. | **Deferred** — re-evaluate when M16 surfaces a concrete requirement. |
| B5 | `ThreadLocalStream` / `new_thread_local_stream` / `set_default_stream` | `vendor/mlx/mlx/stream.h:24-41` | Emily uses `mx::new_stream` per worker thread (`c_src/emily/worker.hpp:102`) and `mx::default_stream(cpu)` for linalg. The newer thread-local APIs are unused. | Could simplify the M14 per-process worker-thread model if MLX's thread-local stream semantics align with Emily's "one stream per BEAM process" guarantee. | Investigative first — may be a no-op if the current model already expresses the same invariant, or a simplification if it doesn't. Low. | **Investigative** — spike to confirm semantics; no code change yet. |

### Re-affirmed out-of-scope (no action)

Catalogued to make the rejection explicit rather than silent:

- **`mlx::distributed::*`** (multi-process collectives, FSDP, ring
  allreduce). PLAN §"Training" ratifies "no distributed training".
  Re-affirm; no change.
- **`mlx::export` / `mlx::import`** (AOT graph serialisation). PLAN
  §Non-goals ratifies "no AOT compilation". Re-affirm.
- **`fast::metal_kernel` / `fast::cuda_kernel`** (user-level GPU kernel
  JIT from Elixir). Orthogonal to Emily's "Nx backend, not a framework"
  stance. Re-affirm.
- **Hadamard transform, einsum path planner, shapeless compile,
  `mlx::event` / `mlx::fence` fine-grained sync.** Noted present;
  no in-roadmap model target currently demands them.
