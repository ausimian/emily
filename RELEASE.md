# Release notes for next release

## Fixed

- Relax MNIST convergence canary threshold from 97% to 96% to eliminate
  stochastic flaps (observed 96.99% on occasional runs). The test is a
  sanity gate, not a performance benchmark.

## Added

- M18 — Observability & fallback telemetry. Makes silent
  `via_binary` round-trips and long-running memory drift observable
  without changing any op semantics.
  - **`Emily.Telemetry`** (`lib/emily/telemetry.ex`) — moduledoc
    enumerates every event, `memory_stats/0` samples the MLX
    allocator and emits `[:emily, :memory, :stats]`.
  - **Span events** via `:telemetry.span/3`:
    `[:emily, :eval, *]` wraps `Emily.eval/1`;
    `[:emily, :to_binary, *]` wraps both `Emily.to_binary/1` and
    `Emily.Backend.to_binary/2` (the Nx.to_binary path) with
    `:shape`, `:dtype`, `:byte_size` metadata;
    `[:emily, :fallback, *]` wraps every `via_binary` / `via_binary_tuple`
    entry with `:op`, `:input_shapes`, `:input_dtypes` metadata.
  - **One-shot fallback warning** per `{op, input_shapes}` pair via
    a `:public, :named_table` ETS dedup set owned by
    `Emily.Application`. **Off by default** — library consumers and
    CI logs stay quiet; opt in with
    `config :emily, :warn_on_fallback, true` (typically in
    `config/dev.exs`) when chasing the Whisper-before-M8 class of
    bug where forward-pass time silently lands on BinaryBackend.
    The telemetry event fires regardless of the config.
  - **Op-name plumbing**: `via_binary/3` → `via_binary/4` and
    `via_binary_tuple/3` → `via_binary_tuple/4` take a leading
    op-name atom; `apply_scatter/7` → `/8` threads it through for
    `indexed_add` / `indexed_put`. All call sites updated.
  - **Tests** (`test/emily/telemetry_test.exs`): fallback start/stop
    event with op+shape metadata (via `Nx.reduce`), 100-call dedup
    capture_log assertion, `warn_on_fallback=false` silence path,
    `to_binary` span metadata with byte_size, and `memory_stats/0`
    emission. `async: false` because the ETS dedup is global.
  - **No `mix.exs` change**: `:telemetry` is already a transitive dep
    via Nx.

- M17 — Conv-pool training (native window ops). Lifted `window_sum`,
  `window_max`, `window_min`, `window_product`, `window_scatter_max`,
  and `window_scatter_min` off the `via_binary` fallback onto native
  MLX. No direct MLX primitives exist — each op is composed as
  `mx::pad` → `mx::as_strided` (sliding-window view) → reduce (for
  the four reductions) or → argmax-with-last-occurrence-tie-break →
  `mx::scatter_add` (for the two scatters). Mirrors MLX's own
  `nn/layers/pooling.py` but generalised to N-D so Nx's per-axis
  window/stride/padding/dilation all lower cleanly.
  - **Closes the training primitive set** — Nx rewrites
    `grad(window_max)` to `window_scatter_max`, so every CNN with
    MaxPool2d lands on the new scatter path on its backward pass. No
    more `via_binary` round-trips during CNN training.
  - **Tie-break** — Nx picks the last-occurrence argmax/min inside a
    window; MLX's `argmax` is first-occurrence. The scatter path
    uses a `mask * arange(K)` trick to recover last-occurrence.
  - **New NIFs** in `c_src/ops/pooling.cpp`: `window_sum`,
    `window_max`, `window_min`, `window_product`,
    `window_scatter_max`, `window_scatter_min`. Corresponding
    stubs in `Emily.Native`; `Emily.Backend` resolves `:valid`/`:same`
    padding and dtype-specific identity values (0/1/±∞ or integer
    min/max) in Elixir before dispatch.
  - **Tests**: `backend_window_test.exs` (19 unit tests —
    reductions × shape × stride × padding × dilation × f32/bf16/s32/u8),
    `backend_window_scatter_test.exs` (11 unit tests — overlapping
    windows + tie-break + non-zero init + 1-D/3-D), grad-equivalence
    extensions for `window_sum`, `window_max` (max-pool grad),
    `window_avg` (3 new zoo fns), `cnn_curve_test.exs` (handwritten
    2-conv + max-pool CNN, 30 steps, per-step loss match), LeNet
    MNIST canary (`:training_full`, target ≥ 97%).

- M16 — Mixed-precision training. `Emily.MixedPrecision` delivers the
  standard bf16 recipe: `cast_params/2` (downcast f32 → bf16 for the
  forward pass), `accumulate_grad/2` (upcast bf16 grads → f32 for the
  optimizer), `loss_scale/1` / `scale_loss/2` / `unscale/2` / `update/2`
  (dynamic loss scaling with overflow detection). `LossScaler` struct
  halves the scale on inf/nan overflow, doubles every N successful steps,
  floors at a configurable minimum. Moduledoc includes a complete
  worked example.
  - **Backend `coerce` fix**: `Emily.Backend.wrap` now checks
    `Native.dtype(ref)` and casts when the MLX buffer dtype disagrees
    with the declared Nx output type. Fixes bf16 grads where
    `Nx.Defn.grad` promotes the output type metadata to f32 but the
    MLX buffer stays bf16.
  - **Tests**: `mixed_precision_test.exs` (33 unit tests), bf16 grad
    equivalence for all 8 zoo functions, bf16 mixed-precision MLP
    curve-matching (50 steps, rtol 5e-2), bf16 MNIST convergence canary
    (`:training_full`, target ≥ 95.5%).

- M15 — Native linalg. `lu`, `svd`, `qr` (reduced), `cholesky`, `eigh`,
  `solve`, and `triangular_solve` now dispatch directly to
  `mx::linalg::*` instead of round-tripping through `Nx.BinaryBackend`.
  MLX's linalg primitives are CPU-only; the NIFs use a CPU stream
  inside the worker's `run_sync` callback.
  - **`qr` with `mode: :complete`** falls back to `via_binary` (MLX only
    supports reduced QR). **`determinant`** uses Nx's default
    implementation, which calls the now-native `lu`.
  - `triangular_solve` handles `left_side: false` and
    `transform_a: :transpose` by composing native transpose + native
    solve (no BinaryBackend fallback).
  - SVD reduced mode (`full_matrices?: false`) slices the full MLX
    result to the target shape.
  - **New NIF stubs** in `Emily.Native`: `linalg_lu/2`, `linalg_svd/2`,
    `linalg_qr/2`, `linalg_cholesky/3`, `linalg_eigh/3`,
    `linalg_solve/3`, `linalg_solve_triangular/4`.
  - Property tests compare all native linalg ops against
    `Nx.BinaryBackend` with well-conditioned random inputs.

- M14.5 — Worker-thread dispatch for vendored MLX. Replaces the
  stream-index NIF convention (M14) and the `safe_eval` mutex with a
  proper per-stream dedicated OS thread. Each `WorkerThread` (C++ class
  in `c_src/emily/worker.hpp`) owns an MLX stream and its Metal
  `CommandEncoder` on a single thread; NIFs dispatch work via
  `run_sync` (promise/future, blocks caller for ~1-10 µs). This
  eliminates the thread-local `CommandEncoder` mismatch that caused
  SIGABRT/SIGSEGV under BEAM process migration, and removes the global
  eval mutex that serialised all GPU work.
  - **MLX built from source.** MLX is vendored as a git submodule
    (`vendor/mlx`, pinned to commit `8e649be4`). The Makefile builds
    `libmlx.a` via cmake and statically links it into the NIF. The
    Metal shader library (`mlx.metallib`) is staged into `priv/` at
    compile time. No prebuilt download step.
  - **`Emily.MlxStream`** GenServer owns the default `WorkerThread`
    resource under the application supervisor. `default_worker/0`
    caches the worker ref in the process dictionary to avoid per-op
    GenServer calls. User-created streams (`Emily.Stream.new/1`)
    allocate their own worker; `enif_monitor_process` stops the thread
    when the creating process exits.
  - **NIF signature change**: every op NIF takes a worker ref as its
    first parameter (after env) instead of a stream index as its last.
    `Emily.Native`, `Emily.Backend`, and all test files updated
    accordingly. `c_src/stream.cpp` (old stream management NIFs)
    deleted.
  - **Concurrent Metal dispatch.** Multiple `WorkerThread`s =
    multiple MLX streams = concurrent Metal command queues. Model
    weights (`mx::array`) are shared across threads (refcounted,
    thread-safe reads). Shared data must be materialised before
    passing across workers (lazy tensors are bound to their creating
    worker's stream).
  - **New files**: `c_src/emily/worker.hpp` (WorkerThread class),
    `c_src/worker_nif.cpp` (resource registration with `down`
    callback), `lib/emily/mlx_stream.ex` (GenServer),
    `test/soak/eval_concurrency_test.exs`.
  - **`Emily.Stream`** rewritten: struct holds a `worker` ref instead
    of an integer index; `with_stream/2` stores it in the process
    dictionary under `:emily_worker`; `synchronize/1` removed.

- M13 — EXLA gradient conformance. Adds a third gradient oracle —
  EXLA (XLA CPU backend) — to catch bugs where Emily and BinaryBackend
  agree on the wrong gradient (they share the same `Nx.Defn.grad`
  lowering). Eight zoo functions plus a full transformer-block training
  step (forward + grad + SGD update) are tested against checked-in EXLA
  golden values. Per-function tolerance tables (linear ops at 1e-6,
  compositions at 1e-4) are calibrated against EXLA 0.11.0 CPU output.
  - **New files**: `test/support/grad_zoo.ex` (shared defn functions),
    `test/support/exla_golden_data.ex` (golden values),
    `test/emily/grad/exla_oracle_test.exs` (test harness),
    `bench/exla_golden_gen.exs` (standalone golden generator script).
  - **Refactored**: `grad_equivalence_test.exs` and
    `finite_diff_test.exs` now import shared functions from
    `Emily.GradZoo` instead of defining inline copies.
  - CUDA conformance deferred to post-1.0.

- M12 — Zero-copy `to_binary`. `Emily.to_binary/1` (and everything
  that routes through `Nx.to_binary` on the Emily backend) now
  returns a BEAM resource binary that aliases the MLX buffer
  directly, instead of memcpying the bytes into a fresh BEAM binary.
  The resource binary's lifetime pins a fresh `Tensor` resource so
  the underlying MLX storage survives until the binary is GC'd.
  Savings are most visible when handing large tensors back to Nx
  (logits from inference, weight exports): one memcpy eliminated
  per call.
  - **NIF change** (`c_src/emily_nif.cpp`): `to_binary` now returns
    `fine::Term` via `fine::make_resource_binary`. Defensive assert
    on `row_contiguous` after `mx::contiguous` guards against
    aliasing a strided buffer.
  - **`from_binary` unchanged.** The memcpy is a one-time cost at
    model load. BEAM → MLX zero-copy via
    `MTL::Device::newBufferWithBytesNoCopy` was investigated and
    dropped (M12.5) — real-world binaries from safetensors never
    meet the page-alignment preconditions.
  - **Tests**: round-trip lifetime test at `test/emily_test.exs`
    (`"to_binary aliased binary survives after tensor goes out of
    scope"`); zero-copy memory soak at
    `test/soak/zero_copy_roundtrip_test.exs` (asserts MLX active
    memory and BEAM binary heap both stay flat across 200 round
    trips). M2 property suite still green — semantics unchanged.
  - **Build**: `EMILY_ASAN=1` env var enables an AddressSanitizer
    build of the NIF (`Makefile`). Requires an OTP built with
    `--enable-sanitizers=address` so beam.smp links the ASan runtime
    at startup — macOS SIP strips `DYLD_INSERT_LIBRARIES` from
    processes launched through `/bin/sh` (which `erl`/`elixir` use),
    and loading libasan late (via dlopen of the NIF) fails because
    the malloc/free interceptors must be installed before any
    allocation. With a sanitizer-enabled OTP:
    ```
    EMILY_ASAN=1 mix compile --force
    ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 mix test
    ```
    No `DYLD_INSERT_LIBRARIES` or `ERL_FLAGS` needed; beam.smp
    already has the runtime linked. A CI job for this is deferred
    until the custom OTP build cost is justified; the lifecycle
    and soak tests provide empirical refcount-safety coverage.
  - **Behavioral change**: `to_binary` returns a resource binary
    aliasing MLX storage. BEAM's binary-vheap GC heuristics don't
    account for the aliased external data (the ProcBin is ~64 bytes
    regardless of the underlying MLX buffer size). In tight loops
    calling `to_binary`, resource binaries can accumulate without
    triggering collection, holding MLX memory longer than expected.
    Callers in hot paths should trigger
    `:erlang.garbage_collect/0` periodically or ensure the binary
    escapes to a short-lived process where GC runs naturally.

- M11 — MLX fused transformer kernels. Wires MLX's handwritten
  `mx::fast::*` fused kernels (RMSNorm, LayerNorm, RoPE, scaled-dot-
  product attention) into Emily as `defn`-callable helpers, and ships
  a Bumblebee shim that swaps these in for the stock composed-defn
  implementations when the Axon graph is rewritten with
  `Emily.Bumblebee.FastKernels.apply/1`.
  - **Native NIFs** (`c_src/ops/fast.cpp`) over `mx::fast::rms_norm`,
    `mx::fast::layer_norm`, `mx::fast::rope`, and
    `mx::fast::scaled_dot_product_attention`. Nullable weight / bias
    / freqs arguments marshal via `std::optional`; the SDPA mask
    argument list marshals via `std::vector<fine::ResourcePtr<Tensor>>`.
  - **`Emily.Fast`** (`lib/emily/fast.ex`) — `rms_norm/3`,
    `layer_norm/4`, `rope/3`, `rope_with_freqs/4`,
    `scaled_dot_product_attention/4`,
    `scaled_dot_product_attention_with_mask/5`. Each helper emits a
    `Nx.Defn.Expr.optional/3` node whose op name matches a custom
    callback on `Emily.Backend`; the Evaluator dispatches to the
    fused kernel under Emily and falls back to a defn composition on
    any other backend. This makes the helpers safe to drop into
    Bumblebee inference paths without breaking BinaryBackend
    conformance runs.
  - **`Emily.Backend.fast_*`** — six custom callbacks (not part of the
    `Nx.Backend` behaviour) that Evaluator picks up when the input
    tensors carry Emily data. They unwrap refs and call the Native
    NIF directly.
  - **`Emily.Bumblebee.FastKernels`** (`test/support/`) — Axon graph
    rewriter, mirroring the M10.5 `Emily.Quantization.Transform`
    pattern. Rewrites `:rms_norm` and `:layer_norm` nodes via
    `Axon.map_nodes`, `Bumblebee.Layers.apply_rotary_embedding/5` by
    function-reference match, and coalesces
    `attention_weights_impl + attention_output_impl` into one fused
    SDPA layer via `Axon.rewrite_nodes`. RoPE handles all four
    Bumblebee scaling strategies (`:linear`, `:dynamic`, `:longrope`,
    `:llama3`) by precomputing the inverse-frequency table Elixir-
    side and passing it to MLX via the `freqs`-override overload.
  - **Tests**: per-kernel Native/defn/equivalence suites at
    `test/emily/fast/`; shim unit tests at
    `test/emily/bumblebee/fast_kernels_test.exs`; fused-kernel
    variants of every `*_full` conformance suite tagged
    `:fast_kernels_full` (excluded by default like the other
    `*_full` tags). Run explicitly:
    `mix test --only fast_kernels_full`.
  - **Bench**: `bench/qwen3_tokens_per_sec.exs` gains an
    `EMILY_BENCH_FAST_KERNELS=1` mode that benchmarks baseline vs
    fused side-by-side, plus an `EMILY_BENCH_PIN=<multiplier>` flag
    that fails with a non-zero exit when the fused mean throughput
    doesn't clear the multiplier × baseline threshold.

- M10 (partial) — Quantized inference primitives. Exposes MLX's affine
  int4/int8 group-wise quantization at the Native and Elixir levels, plus
  a direct-call helper for eager use. Enough to quantize a dense weight,
  store it packed at rest, and dispatch a fused quantized matmul against
  it in plain Elixir code.
  - **Native `Native.quantize/3`, `Native.dequantize/5`,
    `Native.quantized_matmul/7`** (`c_src/ops/linalg.cpp`) over
    `mx::quantize`, `mx::dequantize`, `mx::quantized_matmul`. `quantize`
    returns a 3-tuple `{w_q, scales, biases}` — first NIF in the tree
    returning a tuple of resources. `quantized_matmul` takes the
    `transpose` flag as an explicit arg rather than the PLAN-spec'd `/6`
    signature: AWQ packed layouts need `transpose=false` while
    fresh-from-dense weights use `transpose=true`, and MLX exposes the
    flag as a required parameter on its C++ API.
  - **`Emily.QuantizedWeight`** (`lib/emily/quantized_weight.ex`) —
    `Nx.Container`-derived struct bundling `{value, scales, biases}`
    with `{group_size, bits, transpose}` metadata (the latter flagged
    `keep:` so the scalars survive container traversal — defn trace,
    `backend_transfer`, parameter-map walks). `from_dense/2` validates
    rank ≥ 2, last-axis divisibility, dtype ∈ {f16, bf16, f32}, and bits
    ∈ {2,3,4,6,8} before dispatch. `to_dense/1` inverses via
    `Native.dequantize`.
  - **`Emily.Quantization.quantized_matmul/2`**
    (`lib/emily/quantization.ex`) — direct-call helper over
    materialized tensors. Extracts refs from the input `%T{}` and the
    three tensors inside a `%QuantizedWeight{}`, dispatches to the
    Native NIF, and rewraps the result. Input dtype must match
    `qw.scales.type` (raises with a targeted error otherwise);
    BinaryBackend inputs are transferred transparently. Used by the
    soak and property tests today; the fused-kernel perf win is
    available here but not yet from defn-traced forward passes (see
    below).
  - **`test/emily/native_test.exs`** (+7 cases) — hand-computed
    quantize-shape checks (`group_size=64, bits=4` → 8 u32/row; bits=8
    → 16 u32/row; smaller group_size → more scale/bias rows),
    `dequantize` round-trip within int4 step tolerance, validation of
    indivisible last axes, and `quantized_matmul` equivalence vs.
    `matmul(x, dequantize(…))` for both `transpose=true` and
    `transpose=false` layouts.
  - **`test/emily/quantization/quantized_weight_test.exs`** (new) —
    `from_dense/2` metadata/shape/dtype assertions, validation-raise
    cases (indivisible axis, unsupported bits, unsupported dtype,
    rank < 2), bits=8-tighter-than-bits=4 property, and an
    `Nx.Container` traversal test confirming `keep:` preserves
    `group_size`/`bits`/`transpose` across `backend_transfer`. Property
    test: random group-shaped weights round-trip `from_dense |>
    to_dense` within a 0.15 tolerance band.
  - **`test/emily/quantization/quantized_matmul_test.exs`** (new) —
    property test for `transpose=true` against a
    `Nx.dot(x, Nx.transpose(to_dense(qw)))` oracle on BinaryBackend;
    explicit `transpose=false` case asserting `Nx.dot(x, to_dense(qw))`
    convention; dtype-mismatch raise test; BinaryBackend-input
    auto-transfer test.
  - **`test/soak/quantized_memory_test.exs`** (`@moduletag :soak`,
    default suite) — 1000-iteration quantized-matmul soak asserting
    active memory returns within 4 MB of baseline after `clear_cache/0`.
    Separate from `memory_test.exs` because quantized inference is
    allocator-pattern different from fp16: packed weights load once
    (not re-quantized per call), and the per-iter activation/output
    budget is smaller.
  - **Scope narrowed from PLAN.md.** PLAN spec'd a `Backend.dot/7`
    dispatch path that inspects the operand struct, plus an Axon
    layer-replacement (`Emily.Quantization.Layers.quantized_dense` +
    `Emily.Quantization.quantize/2`) and a `Qwen/Qwen3-0.6B-AWQ`
    conformance test. Investigation uncovered that `Nx.dot/2` calls
    `Nx.LazyContainer.traverse/3` expecting a single `%T{}` — a
    three-tensor `%QuantizedWeight{}` container raises before reaching
    `Backend.dot/7`. The alternative — an Axon layer op calling
    `Native.quantized_matmul` during forward pass — fails for a
    different reason: Axon layer ops run at `Nx.Defn.jit` trace time
    with `Nx.Defn.Expr` inputs, and `Nx.Defn.Evaluator` has no public
    hook to inject a custom op that isn't a `Nx.Backend` callback.
    Closing the gap requires either (a) a defn-native dequantize built
    from Nx bit primitives (loses the fused kernel), or (b) an
    Emily-specific `Nx.Defn.Compiler` variant that recognizes a
    sentinel `Expr` node and routes to `Native.quantized_matmul`. Both
    are meaningful scope; tracked as M10.5. The Native + container +
    direct-call helper surface shipped in M10 is the substrate that
    either of those approaches will build on.

- M9 — Gradient conformance and training primitives. Makes
  `Nx.Defn.grad` usable on Emily by lifting the three ops that grad
  lands on most heavily off the `via_binary` fallback, and adds the
  test scaffolding to trust a gradient. Training on Emily has been
  technically possible since M2 (grad is symbolic in Elixir and
  lowers to forward ops), but every embedding-style backward was
  round-tripping to `Nx.BinaryBackend`.
  - **Native `Native.gather/4`, `Native.scatter/4`, `Native.scatter_add/4`**
    backing `mx::gather`, `mx::scatter`, `mx::scatter_add`. List-of-
    per-axis-indices form (MLX's native shape) rather than the
    `{N, rank}` tensor Nx passes around — the Backend layer does the
    translation.
  - **`Emily.Backend.gather/4`, `indexed_add/5`, `indexed_put/5`**
    rewired. `gather` retains the single-axis `Native.take` fast path
    (now with an explicit output reshape — fixes a latent shape lie
    exposed when downstream MLX ops inspect the ref shape directly,
    e.g. `Nx.gather` followed by `Nx.dot` in a grad); the multi-axis
    path is native. Shared `apply_scatter/6` helper handles index
    splitting (`split_indices_per_axis/3`) and MLX's updates-shape
    rewrap — Nx ships `{batch ++ non_indexed_dims}` but MLX requires
    `{batch ++ per_axis_slot}` with a length-1 dim at every indexed-
    axis position (`updates_shape_for_scatter/3`). Fallback to
    `via_binary` on shapes outside the covered contract.
  - **Duplicate-index divergence.** MLX `scatter` is parallel and
    unordered on duplicate indices; `Nx.indexed_put` is deterministic
    last-write. Documented in the NIF and Backend; grad property
    generators dedupe. `scatter_add` is commutative so duplicates
    accumulate correctly either way.
  - **`test/emily/native_test.exs`** (+5 cases) — scalar-write,
    partial-axis slice write, and the load-bearing `{B, L, D}` target
    with `axes: [0, 1]` rewrap case.
  - **`test/emily/backend_test.exs`** (+8 tests) — targeted cases for
    all three ops exercising the Backend translation paths end-to-end
    via Nx.
  - **`test/emily/grad/grad_equivalence_test.exs`** — property zoo
    covering sum, dot, reshape∘transpose, broadcast, gather,
    indexed_add, plus two composition cases (gather→dot→softmax and
    a mini-attention block). Each case runs under `compiler:
    Emily.Compiler` on Emily.Backend and `compiler:
    Nx.Defn.Evaluator` on `Nx.BinaryBackend`, asserting the
    grad matches within a grad-scaled tolerance. Also covers the
    M9 PRNG-key-threading risk called out in `PLAN.md`: grad through
    a `Nx.Random.uniform_split`-driven dropout with fixed keys
    produces bit-identical results across repeat runs.
  - **`test/emily/grad/finite_diff_test.exs`** +
    **`test/support/grad_helper.ex`** — finite-difference numerical-
    gradient oracle. Pilot of four ops (`sum`, `dot`, `logsumexp`,
    `sigmoid`) with per-op tolerance tables; the harness catches the
    class of bug where symbolic-grad-on-Emily and symbolic-grad-on-
    BinaryBackend *agree* but are both wrong.
  - **`test/emily/training/mlp_curve_test.exs`** +
    **`test/emily/training/transformer_block_curve_test.exs`** +
    **`test/support/training_helper.ex`** — handwritten MLP and
    single transformer block (attention + FFN + residuals) trained
    with vanilla SGD for 50 steps on a fixed synthetic batch. Two
    tolerance bands asserted in each: per-step loss `rtol = 1e-3`
    (silent-drift canary — MLX parallel reductions diverge from
    BinaryBackend sequential reductions, so strict bit-match would
    be flaky) and final-loss `rtol = 1e-4` (convergence
    correctness). No Axon dep at this tier; the training loop is
    handwritten so a red test points at backend/grad numerics.
  - **`test/soak/training_test.exs`** — 1k-iteration training-loop
    memory soak, `@moduletag :soak` (default suite). Reuses the
    handwritten MLP; asserts active memory returns within 2 MB of
    baseline after `Native.clear_cache/0`. Training exercises a
    different allocator pattern than inference (param–grad pairs,
    activations), hence a dedicated soak alongside
    `memory_test.exs`.
  - **`test/emily/training/mnist_full_test.exs`** — opt-in MNIST
    convergence canary, `@moduletag :training_full` (excluded by
    default; run via `mix test --only training_full`). Loads MNIST
    through `scidata`, trains an Axon MLP on `Emily.Compiler` for 5
    epochs, asserts >97% test accuracy. Catches systemic grad drift
    that curve-matching misses because curve-matching uses
    BinaryBackend as its own oracle — MNIST convergence is an
    independent cross-check against real-world training dynamics.
    Typical wall time ~10 s on Apple Silicon.
  - **`{:scidata, "~> 0.1", only: :test}`** added for MNIST loading.
    Kept test-only; Emily itself has no dataset-loading dep.
  - **`test/test_helper.exs`** — `:training_full` added to the
    default-exclude list, documented alongside the other heavyweight
    tags.
  - **`PLAN.md`** — M9 scope formalized; M10 (conv-pool training)
    and M11 (1.0 release) renumbered.

- M8 — Native `conv`. `Emily.Backend.conv/4` now dispatches directly
  to `Native.conv_general` (already bound to `mlx::core::conv_general`
  since M1) instead of round-tripping through `Nx.BinaryBackend`.
  The previous fallback was correct but CPU-bound — ≥90% of the ViT
  and Whisper forward-pass cost. ViT/Whisper full-checkpoint
  conformance tests drop from tens of seconds to under 2 s as a
  side-effect.
  - **Layout translation.** MLX `conv_general` expects NHWC input and
    OHWI weight and returns NHWC; Nx's canonical layout is NCHW/OIHW.
    The new callback composes the caller's `input_permutation`,
    `kernel_permutation`, and `output_permutation` opts with the
    NCHW↔NHWC and OIHW↔OHWI transposes, applying the inverse of
    `output_permutation` on the way out (Nx delivers it in
    user→canonical form; see
    [`deps/nx/lib/nx/shape.ex:729-735`](https://hexdocs.pm/nx/Nx.Shape.html)).
    Two ordered transposes per tensor rather than one composed
    transpose — MLX's lazy graph fuses them and the step-wise form is
    obviously correct across rank 3, 4, and 5.
  - **Integer-operand coercion.** `Nx.conv` returns a float but does
    not cast its operands; MLX conv is float-only. The backend now
    runs `Native.astype(ref, out.type)` on input and kernel before
    the transpose chain. Same-type astype is elided by MLX.
  - **Remaining fallbacks.** `batch_group_size > 1` has no MLX
    primitive and still routes through `via_binary`. Complex-typed
    conv ditto. Neither appears in the pinned Bumblebee ref.
  - **`test/emily/backend_conv_test.exs`** (new) — oracle suite vs
    `Nx.BinaryBackend`. Covers 1-D / 2-D / 3-D; stride, `:same` /
    `:valid` / explicit asymmetric padding; `kernel_dilation` and
    `input_dilation > 1`; grouped and depthwise conv; all three
    permutation options (independently and combined for an NHWC
    end-to-end caller); and integer-input→float-output coercion.
  - **`test/emily/backend_fallbacks_test.exs`** — removed the
    now-obsolete "conv routes through BinaryBackend" test. Added a
    `batch_group_size > 1` fallback case asserting the rare path
    still matches BinaryBackend.

- M7 — Bumblebee conformance breadth. Two new models across four
  new test suites extend M3 (DistilBERT) and M4 (Qwen3) beyond
  encoder-only/decoder-only text into vision and audio.
  - **`test/emily/conformance/vit_test.exs`**
    (`@moduletag :conformance`) — ports `Bumblebee.Vision.VitTest`
    verbatim: three tiny-random architectures (`:base`,
    `:for_image_classification`, `:for_masked_image_modeling`)
    driven with synthetic pixel input `Nx.broadcast(0.5, {1, 30, 30,
    3})`, asserted against the same PyTorch-produced reference
    slices Bumblebee's own suite pins. First conformance suite to
    exercise the `conv` fallback path in anger.
  - **`test/emily/conformance/vit_full_test.exs`**
    (`@moduletag :vit_full`, excluded from `--only conformance`
    because the checkpoint is ~330 MB) — loads
    `google/vit-base-patch16-224`, runs a forward pass on a
    deterministic constant-gray pixel tensor, asserts a pinned
    leading-5 logits slice plus argmax == 763 (ImageNet class
    "revolver"). Uses synthetic input rather than a checked-in JPEG
    fixture so the repo stays free of binary assets and the
    featurizer doesn't enter the assertion surface. Run with
    `mix test --only vit_full`.
  - **`test/emily/conformance/whisper_test.exs`**
    (`@moduletag :conformance`) — ports `Bumblebee.Audio.WhisperTest`
    verbatim: two tiny-random architectures (`:base`,
    `:for_conditional_generation`) driven with the same
    `Nx.sin(Nx.iota({1, 60, 80}))` mel features and decoder ids,
    asserted against Bumblebee's reference slices. First
    conformance suite to exercise encoder-decoder cross-attention on
    Emily, and the first with strided 1-D conv in the encoder
    frontend.
  - **`test/emily/conformance/whisper_full_test.exs`**
    (`@moduletag :whisper_full`, excluded from `--only conformance`
    because the checkpoint is ~150 MB) — loads
    `openai/whisper-tiny`, runs a forward pass on a synthetic 30-s
    mel window (`sin(iota({1, 3000, 80}) * 0.01)`), asserts pinned
    leading 3×3 logits slice + decoder-last-step argmax. Run with
    `mix test --only whisper_full`.
  - **`test/support/conformance_helper.ex`** — shared `use`-able
    module lifting the `setup_all` backend-swap block and
    `assert_all_close/3` out of DistilBERT, Qwen3, ViT, and Whisper
    suites. Net change before the two new suites was ~zero LOC;
    keeps future conformance additions terse.
  - **`test_helper.exs`** — exclude list extended with `:vit_full`
    and `:whisper_full`. Comment rewritten to document each
    heavyweight tag and its cache footprint.
  - **`PLAN.md`** — renumbered: M7 = conformance breadth (this),
    M8 = native conv, M9 = 1.0 release (was M7). MoE / Mixtral
    tracked as deferred pending upstream Bumblebee support.

## Fixed

- `Emily.Backend.via_binary/via_binary_tuple` — pin the default
  backend to `Nx.BinaryBackend` for the duration of the fallback
  `fun` call. Surfaced when ViT tiny-random exercised `conv`: the
  helpers transferred input tensors correctly, but `Nx.conv`
  constructs a scalar internally (`Nx.pad(t, 0, ...)` builds a
  zero-pad tensor) and that scalar landed on whatever the current
  global default was — `Emily.Backend`, because the conformance
  `setup_all` installs it. BinaryBackend then saw a mixed-backend
  operand list and crashed with a FunctionClauseError on `to_binary`.
  Never surfaced before because `test/emily/backend_fallbacks_test.exs`
  doesn't install Emily as the global default (tensors built with
  `backend: Emily.Backend` opt-in) and every prior Bumblebee suite
  had its hot-path ops off the fallback by M4.

## Changed

- M6 — `mlx::core::compile` wrapping: **dropped** after Phase-1
  de-risk. A pure-C++ microbenchmark on MLX 0.25.1 against an Apple
  Silicon GPU showed the fusion win on a Qwen3-0.6B-shaped transformer
  block is 1.04–1.07× on GPU (below the PLAN's 1.20× gate) and a
  regression on CPU (0.82–0.88×). A sanity workload (pure elementwise
  chain) in the same harness shows the expected 2.78× GPU / 1.47× CPU
  wins, confirming the measurement is trustworthy — the limiting factor
  is that MLX compile doesn't fuse matmul with surrounding elementwise
  ops, and transformer inference is matmul-dominated. The BEAM-
  integrated compile path could not exceed this C++ ceiling, so Phase 2
  and 3 were not built.
  - **`bench/native/compile_microbench.cpp`** — standalone C++
    microbench (hand-written RMSNorm + GQA-lite attention + SwiGLU
    block, plus an 8-op elementwise sanity test). Links against the
    vendored libmlx via the same rpath the NIF uses.
  - **`mix bench.native`** (`lib/mix/tasks/bench.native.ex`) — Mix task
    that invokes the new `bench-native` Makefile target with the same
    env `elixir_make` sets, ensuring the bench uses the project's
    pinned MLX without a second fetch. Supports `--seq`, `--warmup`,
    `--iters` args via `mix bench.native -- <args>`.
  - **`bench-native`** target added to the root `Makefile`, producing
    `$(BUILD_DIR)/compile_microbench`.
  - **`bench/compile_microbench.md`** — full results table +
    reproduction instructions. Retained so the decision can be
    re-measured against future MLX releases without rebuilding the
    harness.
  - **`PLAN.md`** updated: M6 section rewritten to record the drop,
    core design decision #1 and the M5 section footnote updated to
    match.

## Added

- M5 — `Emily.Compiler`, an `Nx.Defn.Compiler` implementation that runs
  `defn` computations on `Emily.Backend`. Wraps `Nx.Defn.Evaluator` after
  validating options and pinning the result backend; the Evaluator
  already walks `Nx.Defn.Expr` in Elixir and dispatches each op via
  `Nx.Shared.list_impl!/1`, which finds `Emily.Backend` whenever the
  operands carry it.
  - **`__to_backend__/1`** returns `{Emily.Backend, [device: …]}` so
    `Nx.Defn.to_backend/1` (consulted by `Nx.Serving` and friends)
    allocates inputs and outputs on Emily rather than the process default
    backend. Honours `:device` opt; defaults to `:gpu`.
  - **`__partitions_options__/1`** pins to a single partition. MLX's
    Metal runtime is not safe for concurrent kernel dispatch from
    multiple OS threads (the same constraint that forces
    `max_cases: 1` in `test_helper.exs`); a multi-partition serving
    would race the driver. `:max_concurrency` is accepted for
    `Nx.Serving` API compatibility but values >1 raise.
  - **No external compile cache.** `__compile__/4` returns a closure
    that captures the walked plan; the closure *is* the cache. Callers
    that want reuse across invocations use `Nx.Defn.compile/3` and hold
    the returned function — Bumblebee / `Nx.Serving` already do this on
    warmup. PLAN.md originally specified an ETS cache keyed by
    `{mfa, input_signature}`; deliberately deviated after accounting
    for the per-call ETS deep-copy cost on a Qwen3-sized expression
    tree. PLAN.md updated to record the rationale.
  - **No `mlx::core::compile` wrapping.** That is M6; lazy evaluation
    at the Backend layer suffices for correctness.
  - **`test/emily/compiler_test.exs`** — callback-contract tests
    (`__to_backend__` device routing, partition pinning, unknown-option
    rejection, `:max_concurrency > 1` refusal); op-equivalence tests
    across elementwise / reduction / shape / linalg / container-output
    paths; control-flow equivalence under `defn` for `while` (the
    construct Qwen3's KV-cache update relies on) and `cond`; a
    `Nx.Defn.compile/3` reuse test confirming the closure executes
    repeatedly without re-walking.
  - **`test/emily/compiler_axon_test.exs`** — the M5 exit criterion. A
    3-layer Axon MLP forward pass under `Emily.Compiler` matches
    `Nx.Defn.Evaluator` on the same backend within float tolerance,
    plus a `Nx.Defn.compile/3` reuse case driving multiple inputs
    through one walk.
  - **`:axon`** added as an explicit `only: :test` dep — already
    transitively available via Bumblebee, but the Axon MLP test reaches
    for it directly and shouldn't be hostage to a Bumblebee dep change.

- M0 scaffold: mix project, MLX 0.25.1 prebuilt fetch pipeline,
  Makefile wiring `fine` + MLX, `Emily.Native` NIF surface for tensor
  round-trip, application supervisor skeleton, smoke test suite.
- M1 — `Emily.Native` op inventory. Shared headers in `c_src/emily/`
  (dtype mapping, Tensor resource, helpers); per-category op files
  under `c_src/ops/`:
  - Creation: `zeros`, `ones`, `full`, `arange`, `eye`.
  - Cast: `astype`.
  - Unary elementwise: `negative`, `abs`, `sign`, `floor`, `ceil`,
    `sqrt`, `rsqrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`,
    trig/inverse-trig/hyperbolic family, `sigmoid`, `erf`, `erfinv`,
    `square`, `reciprocal`, `logical_not`, `bitwise_invert`, `isnan`,
    `isinf`, `isfinite`, `conjugate`, `real`, `imag`, `stop_gradient`,
    `round` (with decimals).
  - Binary elementwise: `add`, `subtract`, `multiply`, `divide`,
    `floor_divide`, `remainder`, `power`, `maximum`, `minimum`,
    `logaddexp`, `arctan2`.
  - Compare: `equal`, `not_equal`, `less`, `less_equal`, `greater`,
    `greater_equal`.
  - Logical: `logical_and`, `logical_or`.
  - Bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`,
    `right_shift`.
  - Reductions (axes + keepdims): `sum`, `mean`, `prod`, `max`, `min`,
    `all`, `any`, `logsumexp`; plus `var`/`std` with `ddof`,
    `argmax`/`argmin`, cumulative `cumsum`/`cumprod`/`cummax`/`cummin`/
    `logcumsumexp`.
  - Shape: `reshape`, `transpose`, `squeeze`, `expand_dims`,
    `broadcast_to`, `concatenate`, `stack`, `flatten`, `tile`,
    `swapaxes`, `pad`, `repeat`.
  - Sort family: `sort`, `argsort`, `partition`, `argpartition`,
    `topk`.
  - Indexing: `slice`, `take`, `where`, `take_along_axis`,
    `put_along_axis`, `scatter_add_axis`.
  - Misc: `clip`, `roll`, `softmax`, `array_equal`.
  - Linalg: `matmul`, `tensordot`, `outer`, `inner`.
  - Convolution: `conv_general` (N-D with asymmetric padding,
    dilation, groups, flip).
  - Random: `random_key`, `random_split`, `random_uniform`,
    `random_normal`, `random_randint`, `random_bernoulli`,
    `random_gumbel`, `random_categorical` — keys passed as optional
    tensor args (nil uses MLX's default key sequence).
  - FFT: `fftn`, `ifftn`, `rfftn`, `irfftn`.
  - Memory: `get_active_memory`, `get_peak_memory`,
    `reset_peak_memory`, `get_cache_memory`, `clear_cache` — exposed
    so the soak harness can observe allocator state.
- `Emily.Native.to_binary/1` routes through `mx::contiguous` so
  strided views (transpose/slice/swapaxes/broadcast) materialize
  correctly.
- `test/support/tensor_helpers.ex` — shared build/inspect helpers.
- `test/soak/memory_test.exs` (`@tag :soak`, excluded by default) —
  5000-iteration allocate/eval/drop loop; asserts MLX active memory
  returns within 1 MB of baseline after `clear_cache`.
- `test/emily/dtype_matrix_test.exs` — smoke matrix covering every
  supported dtype across creation, cast, unary (float + numeric),
  binary, reductions, and comparisons.
- Makefile compiles `c_src/**/*.cpp` recursively.

- M2 — `Emily.Backend`, the `Nx.Backend` implementation. Wraps every
  required callback with a thin `Emily.Native` delegation, so any Nx
  computation can run on MLX via
  `Nx.global_default_backend(Emily.Backend)` or `backend:` opts.
  - Creation, cast, unary, binary, shape, indexing, reductions,
    cumulative reductions, sort family, dot, FFT, top_k, take,
    take_along_axis, all_close — all routed directly to MLX NIFs.
  - Compositions where no single MLX primitive exists: `erfc` (`1 -
    erf`), `cbrt` (`sign(x) * |x|^(1/3)`), `logical_xor` (xor of
    boolean-casted operands), `reverse` (take with reversed indices
    per axis).
  - BinaryBackend round-trip fallback for ops that need non-trivial
    composition in v1: `conv`, multi-axis `gather`, `put_slice`,
    batched `dot`, `reduce`/`window_reduce`, `window_sum`/`_max`/etc.,
    `window_scatter_*`, `indexed_add`/`put`, and advanced linalg
    (`lu`, `svd`, `triangular_solve`). Correct but slow; direct MLX
    paths land incrementally as downstream consumers need them.
  - Hard error on `{:f, 64}` (Metal has no f64) and on `bitcast`,
    `from_pointer`/`to_pointer`, `population_count`,
    `count_leading_zeros` — no MLX primitive.
  - Scalar-on-foreign-backend handling: any tensor the callback
    receives that isn't on `Emily.Backend` (Nx routinely passes
    scalars on `Nx.BinaryBackend`) is transferred in transparently.
  - u8↔pred coercion: MLX comparison/logical ops yield `mx::bool_`;
    Nx expects `{:u, 8}`. Any callback whose declared output dtype is
    `{:u, 8}` but whose MLX result is `pred` is cast at the wrap
    boundary.
  - `test/support/backend_generators.ex` — StreamData generators for
    shape, dtype, and tensor values; `assert_close/3` with
    dtype-aware tolerance.
  - `test/emily/backend_test.exs` — property-based oracle tests vs.
    `Nx.BinaryBackend` across creation, cast, every unary/binary,
    shape, indexing, reductions, sort, and dot.
  - `test/emily/backend_lifecycle_test.exs` — init/from_binary/
    to_binary/backend_copy/backend_transfer/inspect/to_batched/bitcast
    raisers.
  - `test/soak/backend_soak_test.exs` — `@tag :soak` 500-iteration
    MLP forward pass; asserts MLX active memory returns to baseline.
  - `test/soak/backend_concurrency_test.exs` — `@tag :soak`
    cross-process determinism check. Runs workers sequentially
    (`max_concurrency: 1`) because MLX's Metal runtime is not
    safe for concurrent kernel dispatch from multiple OS threads;
    the limitation is upstream and documented in the test moduledoc.
- Interior-axis cumulative reductions (`cumulative_sum` and friends
  with `axis: i` where `i != rank - 1`) route through BinaryBackend.
  MLX's cumulative kernels raise "Unable to safely factor shape" on
  several 4-D-and-up view patterns — both the straight call and a
  transpose-to-last-axis workaround hit the same factoring path. The
  last-axis fast path stays on MLX; interior-axis usage is rare on
  our M3/M4 critical path (transformer inference doesn't need it).

- M4 — Qwen3 inference. `Qwen/Qwen3-0.6B` greedy-decodes end-to-end on
  `Emily.Backend` through Bumblebee's causal-LM serving. Everything on
  Qwen3's critical path (QK-norm, rotary embeddings, GQA, SwiGLU FFN,
  RMSNorm, tied embeddings, KV-cache `put_slice` in a `defn` while
  loop) runs correctly.
  - **Native `put_slice/4`** in `Emily.Backend`, backed by a new
    `Native.slice_update/3` NIF over `mx::slice_update`. Replaces the
    BinaryBackend round-trip — autoregressive decoding calls
    `put_slice` per layer per token to append into the KV cache, and
    the old fallback transferred ~1 MB of cache state through the
    allocator on every call. Also fixes a latent bug in the old
    implementation: dynamic scalar-tensor `start_indices` on
    `Emily.Backend` used to slip through unconverted and crash inside
    BinaryBackend. `slice_start` is now applied to every start
    index, matching the `slice/5` callback.
  - **Operand-type promotion in `put_slice`.** `Nx.put_slice`
    promotes the output type across tensor/update (an s32 pad buffer
    clashing with an s64 decoder input becomes s64), but the backend
    callback still receives the original-type operands. We cast both
    `t` and `slice` to `out.type` via `Native.astype` before
    dispatching to `slice_update`. Without this the MLX buffer
    silently disagrees with the Nx shape metadata — the first symptom
    is `Nx.to_binary` returning a half-sized binary and
    `BinaryBackend.bitstring_part` raising a match error deep inside
    the tokenizer decode. Mirrors the arithmetic-op promotion fix
    landed in M3.
  - **`test/emily/conformance/qwen3_test.exs`**
    (`@moduletag :conformance`) — ports `Bumblebee.Text.Qwen3Test`
    verbatim (three architectures: `:base`,
    `:for_causal_language_modeling`, `:for_sequence_classification`),
    with HF reference slices checked in, plus a `greedy generation`
    describe block that drives
    `Bumblebee.Text.Generation.build_generate` on the tiny-random
    causal LM. That smoke test feeds synthetic `input_ids` in
    `[0, 1024)` (tokenizer vocab is 151 k but the tiny checkpoint's
    embedding is 1024 rows), greedy-decodes 16 tokens through the
    full generation pipeline (`Axon.predict` + logit processing +
    `Nx.argmax` + `put_slice` KV-cache update + `defn while`), and
    asserts bit-exact equality against both `Nx.BinaryBackend` run
    on the same inputs *and* a checked-in 16-token reference.
  - **`test/emily/conformance/qwen3_full_test.exs`**
    (`@moduletag :qwen3_full`, excluded from `--only conformance`
    because the checkpoint is ~1.5 GB) — loads `Qwen/Qwen3-0.6B`
    proper, greedy-decodes 32 tokens from a fixed prompt through
    `Nx.Serving`, and asserts the completion string matches a
    checked-in reference. Run with `mix test --only qwen3_full`.
  - **`bench/qwen3_tokens_per_sec.exs`** — standalone wall-clock
    throughput harness. Loads `Qwen/Qwen3-0.6B`, runs N warmup
    iterations + M measured iterations of greedy decode, reports
    tokens/sec. Prompt, token count, and iteration counts are
    overridable via `EMILY_BENCH_*` env vars. Baseline observed on a
    dev M3 host: ~13.8 tok/s at 16 new tokens under the
    `Nx.Defn.Evaluator` compiler (no `mlx::core::compile` wrap yet —
    that lands in M6). Intended as a regression gate, not a headline
    number.
  - **Bumblebee dependency** bumped from Hex 0.6.3 to a pinned `main`
    commit (`273805e9…`) so `Bumblebee.Text.Qwen3` is available — the
    text port is on main but not yet in a Hex release. Revert to a
    Hex version as soon as one ships Qwen3 support.
  - **`test_helper.exs`** extended the exclude list with
    `:qwen3_full` so the weights-heavy test stays out of
    `mix test --only conformance`.

- M3 — DistilBERT end-to-end on Bumblebee. Every Nx op on the
  transformer critical path now runs natively on MLX; the full
  forward pass matches HuggingFace Transformers (PyTorch) reference
  values within f32 tolerance.
  - **Native batched `dot/7`** in `Emily.Backend`, replacing the
    BinaryBackend bounce. Permutes operands to
    `[batch… , free… , contract…]`/`[batch…, contract…, free…]`,
    collapses to 3-D, dispatches to `Native.matmul` (which treats
    leading dims as batch), reshapes to Nx's canonical
    `batch ++ free_a ++ free_b` layout. Hits 12× per DistilBERT
    forward pass (2× per attention layer × 6 layers). Falls back to
    BinaryBackend for non-float dtypes — MLX matmul is float-only.
  - **Binary op type promotion** fixed at the Backend boundary.
    MLX's cross-type promotion for mixed integer widths (e.g.
    `right_shift(u64, s32)`) falls to float32 and then rejects the
    op. `Emily.Backend` now casts both operands to the Nx-computed
    output type (for arithmetic/bitwise) or merged input type (for
    compare/logical) before dispatching to MLX. Unblocks
    `Nx.Random.key`, which is pulled in transitively even in
    inference-only models via Axon's dropout defn.
  - **Dynamic `slice` starts.** Nx passes scalar-tensor starts under
    `defn` evaluation; `Emily.Backend.slice` now materialises them
    to their concrete values on the fly.
  - **`bitcast`** implemented via `mx::view` (zero-copy reinterpret
    cast between equal-width dtypes). Required by `Nx.Random` to
    move between f32 and u32 bit patterns.
  - **`argmax`/`argmin` keep-axis robustness.** Derive the keep-axis
    flag from `out.shape` vs input rank instead of trusting the raw
    opts key (Nx's user-facing API uses `:keep_axis`, singular, while
    some callers pass `:keep_axes`).
  - **`test/emily/conformance/distilbert_test.exs`**
    (`@moduletag :conformance`, excluded by default; run with
    `mix test --only conformance`) — ports Bumblebee's own DistilBERT
    tests verbatim. Six architecture variants (`:base`,
    `:for_masked_language_modeling`,
    `:for_sequence_classification`, `:for_token_classification`,
    `:for_question_answering`, `:for_multiple_choice`) plus an
    `Nx.Serving.batched_run` smoke test exercising the QA pipeline
    end-to-end (tokenizer → model → postprocess).
  - **CI runs the conformance suite** on every push/PR as a separate
    step after `mix precommit`. `~/Library/Caches/bumblebee` is
    cached across runs so the ~3 MB HF fixture download happens
    once. Local `mix test` remains opt-in via `--only conformance`
    so a fresh-clone/offline contributor isn't blocked by network.
  - **Batched-dot property tests** added to
    `test/emily/backend_test.exs` — 1- and 2-axis batch cases plus
    edge shapes (scalar output, multi-free-axis both sides).
  - **Test-only deps:** `bumblebee ~> 0.6`, `tokenizers ~> 0.5`
    (both `only: :test`). Nx pinned to `~> 0.10` (down from 0.11)
    to match Bumblebee's current constraint; emily's own API is
    unaffected.

## Fixed

- `Emily.Backend.put_slice/4` — swapped `slice` and `start_indices`
  parameters. Latent since M2 because the callback routes through
  the BinaryBackend fallback and had no direct test. Surfaced by
  the new fallback-coverage suite.

## Tests

- `test/emily/backend_fallbacks_test.exs` — smoke coverage for every
  `via_binary` branch (`put_slice`, multi-axis `gather`, `conv`,
  `reduce`, `window_reduce`, `window_sum`/`_product`/`_max`/`_min`,
  `window_scatter_max`/`_min`, `indexed_add`/`_put`, `lu`,
  `triangular_solve`, `svd`) plus the forced-fallback branches
  (integer batched `dot`, interior-axis `cumulative_*`). The
  fallback dispatches to BinaryBackend, so comparing against
  BinaryBackend is tautological — these tests verify the transfer /
  compute / rewrap round-trip runs clean, not numerical correctness.
- Extended `test/emily/backend_lifecycle_test.exs` with the three
  raise-only callbacks (`count_leading_zeros`, `population_count`,
  `pad` with interior padding), the `backend_transfer(t, Nx.Tensor)`
  identity case, the `from_binary` iodata path, and the
  `inspect` `:infinity` limit branch.
- Aggregate coverage with `mix test --cover --include conformance`:
  74.7% → 81.9% total; `Emily.Backend` 73.5% → 82.3%. Remaining
  uncovered in `Emily.Backend` is a handful of functional ops not
  yet in the property suite (`fft`/`ifft`/`fft2`/`ifft2`, `argsort`,
  `top_k`, `erfc`, `cbrt`, `all_close` with `equal_nan: true`) plus
  unreachable defensive branches.

## Notes

- Ops files use anonymous namespaces to prevent NIF function names
  (`sin`, `log1p`, `sqrt`, ...) from colliding with C math-library
  symbols pulled in by MLX headers.
- Deferred beyond M1: the full `scatter`/`scatter_add`/... family with
  vector-of-indices (only the axis-aligned forms are bound),
  `hadamard_transform`, quantized matmul, `linalg.*` decompositions
  (LU, QR, Cholesky, SVD). These will be added opportunistically when
  M2/M3 callers need them.
- Deferred beyond M3: native `conv` translation (PLAN lists it under
  M3, but DistilBERT and M4's Qwen3 don't use it; the BinaryBackend
  fallback remains until a CV model lands on Emily).
