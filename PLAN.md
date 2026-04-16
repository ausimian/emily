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

The eleven milestones below were derived from a structured review of
the post-M9 codebase (see PR discussion) and ordered by
user-visible value, not difficulty. Headline rationale:

1. **Quantization, fast kernels, zero-copy** (M10–M12) move the needle
   most for the headline use case (Bumblebee Qwen3 on a MacBook).
2. **Grad oracle, serving, linalg** (M13–M15) close correctness and
   production-readiness gaps that block real adoption.
3. **Mixed-precision and conv-pool training** (M16–M17) finish the
   training story M9 started.
4. **Observability, errors, interop, doctor** (M18–M21) are the polish
   that turns a working library into one new users can actually adopt
   without hand-holding.
5. **1.0 release** (M22) ships the result.

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

### M19 — Error surfacing

C++ exceptions propagate through `fine` and surface as
`{:error, "raw MLX message"}` with no Elixir context. "Unable to
safely factor shape" and "binary size mismatch" are common MLX
strings but unhelpful as-is.

- Wrap each NIF entry to catch known MLX exception classes
  (`std::invalid_argument`, `std::runtime_error`, MLX-specific) and
  re-raise as tagged Elixir exceptions: `Emily.ShapeError`,
  `Emily.DtypeError`, `Emily.MLXError`.
- Each exception carries the op name, input shapes/dtypes, and the
  raw MLX message. The Backend layer optionally annotates with the
  originating Nx callback name.
- Migrate existing raises (`f64` rejection, `count_leading_zeros`
  unsupported, etc.) onto the same exception types for consistency.

**Testing**: deliberate failure for each exception class, assert the
raised struct's fields.

**Exit:** all NIFs catch and translate; exception hierarchy
documented.

### M20 — GPU interop pointers

`from_pointer` / `to_pointer` currently raise. Forecloses handing
Emily tensors to user-owned native code (audio/vision pipelines,
custom kernels, future DLPack interop). Lowest-priority of the eleven
gaps — real but narrow.

- `to_pointer`: return the MLX buffer device pointer (Metal for GPU,
  raw for CPU) plus a refcount handle the caller must release.
  Document the lifetime contract loudly.
- `from_pointer`: construct an MLX array from a user-owned pointer
  with a caller-supplied deleter. Same lifetime contract.
- DLPack export/import is the natural follow-on — same shape of
  problem, broader audience. Out of scope for M20.

**Testing**: round-trip via a tiny C harness in `test/c/`; assert
no leak under valgrind.

**Exit:** pointer ops documented with lifetime warnings; no segfaults
in CI.

### M21 — `mix emily.doctor`

The MLX prebuilt fetch + checksum + `elixir_make` chain is what
breaks on fresh macOS machines. A diagnostic Mix task pays back on
day one of adoption.

- Probes: Xcode CLT version, MLX prebuilt presence + checksum, NIF
  load, trivial GPU dispatch, MLX version vs. expected, available
  unified memory.
- Output: a structured report with green/yellow/red per probe;
  remediation hints per failure.

**Testing**: snapshot test of the report on a known-good
configuration; assert each probe surfaces a useful message on a
deliberately-broken configuration (rename the prebuilt directory,
revoke read on the cache, etc.).

**Exit:** task documented in the README setup section; surfaces clear
errors for the three most common breakages.

### M22 — 1.0 release (was M11)

- API docs, HexDocs, README with worked Bumblebee + quantized-Qwen3
  examples
- Hex release (public), versioned per conventions (`@version` in mix.exs)
- `RELEASE.md` accumulated across feature branches

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
