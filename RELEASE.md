### Added

- **Single-NIF native compiler — `Nx.Defn.jit`/`compile` with
  `compiler: Emily.Compiler, native: true`.** A real `Nx.Defn.Compiler` path
  that lowers a traced `Nx.Defn.Expr` to a flat IR **once** and replays the
  whole forward graph in a **single NIF call per invocation**, instead of one
  BEAM↔worker round-trip per op. For dispatch-bound workloads — autoregressive
  decode, where the structurally-identical graph is otherwise rebuilt op-by-op
  every token — this collapses the per-token dispatch cost (a 100-op microbench
  shows a >15× build/dispatch collapse). Weights cross the NIF boundary once
  (captured by the compiled program) and are never re-serialized per call.
  Opt in per call:

      Nx.Defn.jit(&forward/1, compiler: Emily.Compiler, native: true).(input)

  Coverage is **no-fallback**: the full Nx primitive set (with `Emily.Backend`'s
  dtype-coercion and op-composition semantics ported into the lowering), the
  fused `Emily.Fast.*` / `Nx.Block.*` kernels (RMSNorm, LayerNorm, RoPE, scaled
  dot-product attention and its mask/sink variants, the LinAlg blocks),
  quantized matmul (now an `Nx.block` node so it fuses under the compiler too),
  dynamic KV-cache writes (`put_slice` at a runtime offset), container/tuple
  outputs, and `cond` (lowered to a select-chain). DistilBERT and ViT forwards
  run end-to-end under the compiler with `config :emily, :fallback, :raise`.
  Constructs the IR can't lower yet — `while` loops, arbitrary BEAM `reduce`
  functions — are handled by the graceful fallback below.

  An opt-in compiled eval mode additionally wraps the replay in
  `mlx::core::compile`, fusing the elementwise runs (rms-norm, softmax, SiLU
  gating, residual adds) the replay leaves as separate kernels — measured at
  ~1.5–1.6× over the plain replay on a decode-shaped transformer block.

- **Graceful native fallback — `native_fallback: :eval` (the default).** When a
  `native: true` defn contains an op or construct the Expr compiler can't lower
  yet, the *whole* defn now routes through `Nx.Defn.Evaluator` (each op then
  dispatches through `Emily.Backend`, with its own per-op fallback) and emits a
  one-shot `[:emily, :compiler, :fallback]` telemetry event — instead of
  raising. This makes it safe to install the compiler globally on any model:

      Nx.Defn.global_default_options(compiler: Emily.Compiler, native: true)

  Covered subgraphs (e.g. encoder forwards) run fully native; the rest is
  transparently evaluated. Pass `native_fallback: :raise` (or set
  `config :emily, native_fallback: :raise`) to fail instead — the conformance
  suites use this to prove a model lowers fully native.

- **More ops lower natively** — `argmax`/`argmin`, `clip`, and `sort`/`argsort`
  (ascending and descending) now compile under the native single-NIF path
  rather than routing through the fallback, each mirroring its `Emily.Backend`
  callback bit-for-bit. `argmax` in particular puts greedy-decode token
  selection on the native path. Remaining gaps (`gather`/scatter,
  pooling/`window_*`, cumulative) continue to work via the graceful fallback.

- **19 more unary ops lower natively** — `expm1`, `tan`, `sinh`, `cosh`,
  `acos`/`asin`/`atan`, `acosh`/`asinh`/`atanh`, `round`, `bitwise_not`,
  `is_nan`/`is_infinity`, `conjugate`, `real`/`imag` route to the same
  `mx::*` primitive as the eager unary NIF (bit-identical to the
  Evaluator); `erfc` and `cbrt` compose from existing ops, mirroring
  `Emily.Backend`'s eager composition. Closes the largest cluster on the
  Expr op-coverage checklist (#188).

- **`atan2`, `quotient`, `logical_xor` lower natively** — closes the
  binary/compare cluster on the Expr op-coverage checklist (#188).
  `atan2` slots into the @arith_binary mapping (cast both to out.type,
  then `mx::arctan2`); `quotient` routes through `mx::floor_divide` the
  same way `Emily.Backend.quotient/3` does; `logical_xor` (no MLX
  primitive) lowers to `(a != 0) != (b != 0)`, mirroring the eager
  Backend composition. All three are bit-identical to the Evaluator.

- **`pad`, `eye`, `triangular_solve` lower natively** — closes the
  top-level-Expr-op cluster on #188. `pad` is the constant-pad path
  (interior dilation raises, same as `Emily.Backend.pad/4`); `eye`
  materialises as a captured constant (identity matrix at lower time,
  same trick `iota` uses); `triangular_solve` decomposes all four
  `transform_a` × `left_side` combinations into transposes around a
  bare `linalg_solve_triangular` opcode, mirroring
  `Emily.Backend.triangular_solve/4`. The C++ dispatcher routes
  `mx::linalg::solve_triangular` to the CPU stream per call (it's
  CPU-only), matching the eager NIF.

- **`Nx.Block.LogicalNot`, `Nx.Block.AllClose`, `Nx.Block.Phase` lower
  natively** — closes the misc-block cluster on #188. No new opcodes:
  `LogicalNot` emits the existing `:logical_not` op directly (same as
  `Emily.Backend.native_logical_not/2`); `AllClose` composes five
  existing primitives — `astype` → `abs(a - b) <= atol + rtol * abs(b)`
  → optional `isnan` OR for `equal_nan: true` → reduce-all — exactly
  matching `Emily.Backend.native_all_close/4`; `Phase` lowers the
  block's `atan2(imag(t), real(t))` expansion via TopK-style parameter
  seeding (every primitive in the expansion was already on the native
  path), bit-identical to the Evaluator.

- **`Nx.LinAlg.cholesky` / `solve` / `qr` / `eigh` / `lu` / `svd` /
  `determinant` lower natively** — closes the LinAlg-block cluster on
  #188. Two new infrastructure pieces enable this: a small extension to
  `replay_program` that pushes N values per instruction for multi-output
  linalg ops (mirroring how `while` already pushes one value per
  loop-carried state element), and a "shallow" parameter collector that
  stops at nested `:block` nodes so the outer block's seeding doesn't
  leak across scope boundaries (Determinant's expansion calls
  `Nx.LinAlg.lu` internally — without the fix the IR would try to bind
  LU's fresh inner block-params from the outer scope). Cholesky and
  Solve are single-output; QR/Eigh/LU/SVD are multi-output. LU's
  `s32` perm vector is post-processed into a permutation matrix via
  `take(eye(n), perm, 0)` exactly like `Emily.Backend.native_lu/3`; SVD
  in `full_matrices?: false` mode is sliced from the full output the
  same way `Emily.Backend.maybe_slice_svd/4` does (the Gram thin-tall
  workaround for Issue #84 stays on the eager path for now);
  Determinant has no fused kernel — both paths run the block's composed
  2x2 / 3x3 / NxN expansion, via the same TopK-style parameter seeding.
  All seven dispatch to the CPU stream per call (MLX's `linalg::*` is
  CPU-only); the C++ side overrides the replay stream just like the
  eager linalg NIFs.

- **`take_along_axis` lowers natively** — `Nx.take_along_axis` (the
  `Nx.Block.TakeAlongAxis` block) now compiles under the native single-NIF
  path, mirroring `Emily.Backend.native_take_along_axis/4` (cast indices to
  s32, then `mlx::core::take_along_axis`) bit-for-bit. This was the last op
  forcing a fallback in `Bumblebee.Text.question_answering`'s answer-span
  gather, so a DistilBERT question-answering `Nx.Serving` forward now runs
  fully native — and fused — under `native_fallback: :raise`.

- **The FFT family lowers natively** — `Nx.fft`/`ifft` (1-D, trailing axis)
  and the `fft2`/`ifft2`/`rfft`/`irfft` blocks now compile under the native
  single-NIF path instead of falling back. Each mirrors its eager
  `Emily.Backend` wrapper, routing to the same `mlx::core::fft::*` kernel
  (unnormalized `FFTNorm::Backward`) bit-for-bit, including the complex64
  outputs. This was the op forcing a graceful fallback in a Whisper
  `speech_to_text` serving — the log-mel featurizer's STFT — so that path
  now compiles fully native too.

- **`indexed_put` / `indexed_add` (scatter) lower natively** — both now
  compile under the native single-NIF path for MLX-scatter-compatible index
  layouts (the same layout the native gather already requires), mirroring
  `Emily.Backend`'s scatter — split the index tensor into per-axis s32
  arrays, reshape updates into MLX's layout, then `mx::scatter` (overwrite)
  / `mx::scatter_add` (accumulate) bit-for-bit. Layouts MLX can't scatter
  still route through the evaluator under `native_fallback: :eval`.

- **`Nx.top_k` lowers natively.** The `Nx.Block.TopK` block — a multi-output
  `{values, indices}` — now compiles instead of raising. `Emily.Backend` has
  no `top_k` override (`mx::topk` yields values only, not the indices Nx's
  contract requires), so the evaluator computes it via the block's default
  expansion (`argsort(desc)` → `take_along_axis` → `slice` the top k); the
  compiler lowers that same expansion — every op in it already lowers — and
  projects the two leaves via `:elem`, bit-identical to the evaluator.

- **Window (pooling) ops lower natively — forward and backward.** The
  forward window family (`window_sum`/`window_max`/`window_min`/
  `window_product`, i.e. average and max pooling), the select-and-scatter
  backward (`window_scatter_max`/`window_scatter_min`, the MaxPool/MinPool
  gradient), and `reverse` (the conv-backward kernel flip) all now compile
  under the native single-NIF path instead of falling back. The pad →
  sliding-window → reduce/scatter cores moved into `emily/op_cores.hpp` so
  the eager NIFs and the compiled replay share one implementation. A
  small-CNN **training step** (conv + maxpool forward and backward, grad,
  SGD) now lowers fully native under `native_fallback: :raise`, producing a
  loss bit-identical to the evaluator. Native training is now
  **convergence**-tested, not just verified-lowering: handwritten CNN
  (30 SGD steps) and MLP (50 steps) trajectories match the op-by-op
  evaluator bit-for-bit and a `BinaryBackend` oracle to f32 tolerance, and
  full **Axon** training drives native end-to-end — `Axon.Loop.run`
  forwards `native: true`/`native_fallback:` to the defn jit, so a LeNet
  CNN and a dense MLP train on real MNIST entirely through the single-NIF
  path (forward, categorical-cross-entropy, backward, Adam) and reach the
  same >97% / >96% accuracy as the evaluator (`:training_full`).

- **`Bumblebee.Text.generation` compiles fully native — greedy and sampling.**
  The headline result: an end-to-end Bumblebee generation (the transformer
  forward, the `defn while` decode loop, dynamic KV-cache writes, `cumsum`
  position ids, `argmax`/multinomial token selection, threefry sampling) now
  lowers to the single-NIF replay with **no fallback**, producing token ids
  bit-identical to the Evaluator. Wiring this required `cumsum`/`cumprod`/
  `cummax`/`cummin` (last-axis fast path), single- and multi-axis `gather`,
  and `stack`. Greedy and multinomial sampling are gated in
  `generation_native_test.exs`. On Qwen3-0.6B this measures **~5× the
  evaluator's decode throughput** (~61 vs ~12 tok/s on an M-series Mac),
  with byte-identical completions — the single-NIF replay collapses the
  per-op BEAM↔worker round-trips to roughly one per token. Reproduce with
  `bench/qwen3_tokens_per_sec.exs` (baseline vs native lanes).

- **Fused-while decode — `fuse: true`.** An opt-in lane that fuses
  the `defn while` decode loop's body under `mlx::core::compile`. The outer
  loop is data-dependent and host-controlled, so `mx::compile` can't trace it
  as a whole — but the per-token forward (the loop *body*) is shape-stable
  (`offset` is a runtime input), so each iteration replays through a
  per-stream-cached compiled callable that fuses the elementwise runs the plain
  replay leaves separate (RMSNorm, softmax, SiLU gating, residual adds). The
  compiled body cache-hits across tokens rather than recompiling per step.
  Enable it on top of the native generation path:

      Nx.Defn.jit(&forward/1,
        compiler: Emily.Compiler, native: true, fuse: true)

  On Qwen3-0.6B this lifts greedy decode to **~5.4× the evaluator (~1.1× over
  the plain native lane**, ~68 vs ~62 tok/s on an M-series Mac). The trade-off:
  `mx::compile` reassociates f32, so logits drift by a few ULP and the output
  is **not** bit-identical to the evaluator. Greedy argmax is robust to that
  drift, so in our Qwen3-0.6B run the generated token ids matched the
  evaluator's exactly (byte-identical completions) — but that is an *empirical*
  result, not a guarantee: a near-tie top-2 logit can flip a token, and any
  decision the drift can tip over — argmax, or a loop trip count whose
  condition reads a reassociated reduction — diverges by more than a few ULP
  once it flips. **Sampling strategies (e.g. multinomial) will diverge from the
  evaluator under fusion** even with a fixed seed; the gate and bench cover the
  greedy lane only. The `native-fused` lane in `bench/qwen3_tokens_per_sec.exs`
  measures throughput; `generation_native_test.exs` gates the greedy token
  match.

- **`defn while` compiles native.** Data-dependent loops — including
  `Bumblebee.Text.generation`'s decode loop — now lower to the single-NIF
  replay instead of falling back. The condition and body become nested
  sub-programs (their loop-carried state bound as inputs); the worker thread
  runs the loop, evaluating the condition each iteration to decide whether to
  continue, so the **whole loop is one NIF call** with no per-iteration
  BEAM↔worker round-trip. The `while` instruction is multi-output — its
  outputs are the final loop-carried state — and `:elem` projects them. (The
  opt-in `mx::compile` eval mode can't trace the data-dependent host loop as a
  whole, so it fuses each loop *body* instead — see the fused-while lane
  below.)

- **`Nx.Random` compiles native.** The PRNG surface (`split`, `uniform`,
  `normal`, `randint`, `gumbel`, `choice`) now lowers under the native
  compiler, so sampling-based generation runs on the single-NIF path and a
  PRNG key threads through a decode loop as ordinary carried state. This
  needed three primitives: `bitcast` (random bits → float), `erf_inv`
  (`normal`), and a **dynamic-start `slice`** — threefry indexes its rotation
  table by the loop counter, a genuine runtime-start slice (the eager backend
  materialises the index to a host int; the compiled replay threads it as a
  runtime `s32` instead, same result).

- **`Emily.Generation` — a model-agnostic decode-loop driver.** JIT-compiles a
  caller-supplied **shape-stable** per-token forward (`fn token, offset, cache,
  params -> {logits, cache} end`) with the native single-NIF compiler and drives
  the autoregressive loop from Elixir: offset bookkeeping, KV-cache threading,
  stop conditions, next-token selection (greedy by default), and per-token
  streaming via `:on_token`. The forward runs fully native; the loop stays in
  Elixir, so token streaming and host-side control are preserved. Emily supplies
  only the mechanism — the model (forward + cache) is the caller's.

- `Emily.async_eval/1` (and `Emily.Native.async_eval/2`) schedule evaluation of
  one or more lazy graphs **without blocking on the GPU**, wrapping
  `mlx::core::async_eval`. The work is handed to the device's command queue and
  the call returns as soon as it is enqueued — not when it finishes. This lets a
  caller keep dispatching the next step's ops while the device computes the
  current one (e.g. an autoregressive decode loop), blocking only when a value
  is actually read back on the host via `to_binary/1` / `eval/1`. Pass every
  output of a step (logits plus all KV-cache buffers) in one call.
- `Emily.Native.fast_rope_int/8` — RoPE with an **integer** absolute-position
  `offset` (routing to MLX's int-offset `rope` overload), for incremental decode
  where the caller tracks position host-side. Complements the existing
  tensor-offset `fast_rope/8`. Note: feed the kernel the 4-D
  `{batch, heads, seq, head_dim}` layout — in 3-D, MLX 0.31 mis-rotates
  single-token (`seq == 1`) inputs.

### Fixed

- **A tuple-returning `cond` hard-crashed the native compiler instead of
  lowering.** A `cond`/`if` whose branches return a tuple (multi-output) hit a
  `FunctionClauseError` in the lowerer — and because that isn't an
  `ArgumentError`, it escaped the graceful-fallback rescue and faulted rather
  than degrading to the evaluator. It now lowers to one `where`-chain per leaf
  (same wholesale-select semantics as a single-output `cond`), projected by
  `:elem`; a nested/non-tensor container raises cleanly (graceful fallback).
  Surfaced by a Whisper `speech_to_text` serving — with this, plus the native
  `fft` and `indexed_put` lowering above, the **full Whisper serving
  (featurizer STFT + encoder/decoder + autoregressive decode loop) compiles
  fully native end-to-end**, gated by `whisper_full_test.exs`.

- **Dilated window reductions (`window_dilations > 1`) returned wrong values.**
  `window_sum`/`window_max`/`window_min`/`window_product` with a dilated kernel
  silently produced garbage for windows past the first stride positions, on both
  the eager backend and the native compiler (they share the window-reduce core).
  A dilated kernel axis gets an `as_strided` stride > 1, so the sliding-window
  view aliases fewer physical elements than its logical size; MLX's strided-reduce
  fast path then read past the aliased buffer. The view is now materialised
  contiguously before the reduce when any dilation > 1 (the common non-dilated
  pooling path is unchanged and stays copy-free).
