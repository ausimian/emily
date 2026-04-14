# Release notes for next release

## Added

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
