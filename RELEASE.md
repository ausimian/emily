### Added

- **Native Expr compiler — `Nx.Defn.jit` / `compile` with
  `compiler: Emily.Compiler, native: true`.** Lowers a traced
  `Nx.Defn.Expr` to a flat IR once and replays the whole forward graph
  in a **single NIF call per invocation**, collapsing the per-op
  BEAM↔worker round-trips a step-evaluated decode loop would otherwise
  pay. Weights cross the NIF boundary once (captured by the compiled
  program) and are never re-serialised per call. Opt in per call:

      Nx.Defn.jit(&forward/1, compiler: Emily.Compiler, native: true).(input)

  Coverage is the full Nx primitive set (with `Emily.Backend`'s
  dtype-coercion and op-composition semantics ported into the
  lowering), the fused `Emily.Fast.*` kernels (RMSNorm, LayerNorm,
  RoPE, scaled dot-product attention and its mask / sink / mask+sink
  variants), `Nx.Block.*` including the full `LinAlg` family
  (`cholesky` / `solve` / `qr` / `eigh` / `lu` / `svd` /
  `determinant`), `Nx.Random`, and the control flow `cond` /
  `defn while` (with the host loop driven entirely from the worker
  thread). Anything the IR can't lower yet routes through
  `Nx.Defn.Evaluator` under the default `native_fallback: :eval` (with
  a one-shot `[:emily, :compiler, :fallback]` telemetry event), so
  installing the compiler globally is safe on any model:

      Nx.Defn.global_default_options(compiler: Emily.Compiler, native: true)

  The `native` default is also read from `config :emily, :native`
  (defaulting to `false`), so `config :emily, native: true` opts every
  defn into the native lane application-wide without a per-call option;
  a per-call `native:` option always wins over the app-env default.

  `native_fallback: :raise` fails instead — the conformance suites use
  this to prove a model lowers fully native.

  End-to-end: DistilBERT (question answering with `Nx.Serving`), ViT,
  Whisper (`speech_to_text` end-to-end including the featurizer STFT,
  encoder/decoder, and autoregressive decode loop), and Bumblebee
  `Text.generation` (greedy *and* multinomial sampling) all compile
  fully native under `native_fallback: :raise`. Bumblebee generation
  on Qwen3-0.6B measures **~5× the evaluator's decode throughput**
  (~61 vs ~12 tok/s on an M-series Mac), with byte-identical
  completions. Native training drives Axon end-to-end — a LeNet CNN
  and a dense MLP train on real MNIST entirely through the single-NIF
  path (forward, categorical-cross-entropy, backward, Adam) to the
  same >97% / >96% accuracy as the evaluator.

- **`Emily.Compiler` — `:fuse` opt-in.** Adds `mx::compile` fusion on
  top of the replay, fusing elementwise runs (RMSNorm, softmax, SiLU
  gating, residual adds) the plain replay leaves as separate kernels.
  For a `defn while`, the loop body is fused under `mx::compile` and
  cached per stream so it cache-hits across iterations rather than
  recompiling per step. Enable on top of the native generation path:

      Nx.Defn.jit(&forward/1,
        compiler: Emily.Compiler, native: true, fuse: true)

  On Qwen3-0.6B this lifts greedy decode to **~5.4× the evaluator
  (~1.1× over the plain native lane)**, ~68 vs ~62 tok/s; in
  isolation on a decode-shaped transformer block, fusion measures
  ~1.5–1.6× over the plain replay. Trade-off: `mx::compile`
  reassociates f32 to within a few ULP, so output is **not**
  bit-identical to the evaluator. Greedy argmax is robust to that
  empirically (Qwen3-0.6B token ids matched the evaluator exactly in
  our run), but the match is empirical, not guaranteed — a near-tie
  top-2 logit can flip a token. **Sampling strategies will diverge
  from the evaluator under fusion** even with a fixed seed.

- **`Emily.Generation` — a model-agnostic decode-loop driver.**
  JIT-compiles a caller-supplied shape-stable per-token forward
  (`fn token, offset, cache, params -> {logits, cache} end`) with the
  native single-NIF compiler and drives the autoregressive loop from
  Elixir — offset bookkeeping, KV-cache threading, stop conditions,
  next-token selection (greedy by default), and per-token streaming
  via `:on_token`. The forward runs fully native; the loop stays in
  Elixir, so token streaming and host-side control are preserved.
  Emily supplies only the mechanism — the model (forward + cache) is
  the caller's.

- `Emily.async_eval/1` (and `Emily.Native.async_eval/2`) schedule
  evaluation of one or more lazy graphs **without blocking on the
  GPU**, wrapping `mlx::core::async_eval`. The work is handed to the
  device's command queue and the call returns as soon as it is
  enqueued — not when it finishes. Lets a caller keep dispatching the
  next step's ops while the device computes the current one (e.g. an
  autoregressive decode loop), blocking only when a value is actually
  read back on the host via `to_binary/1` / `eval/1`. Pass every
  output of a step (logits plus all KV-cache buffers) in one call.

- `Emily.Native.fast_rope_int/8` — RoPE with an **integer**
  absolute-position `offset` (routing to MLX's int-offset `rope`
  overload), for incremental decode where the caller tracks position
  host-side. Complements the existing tensor-offset `fast_rope/8`.
  Note: feed the kernel the 4-D `{batch, heads, seq, head_dim}`
  layout — in 3-D, MLX 0.31 mis-rotates single-token (`seq == 1`)
  inputs.

### Fixed

- **Dilated window reductions (`window_dilations > 1`) returned wrong
  values.** `window_sum`/`window_max`/`window_min`/`window_product`
  with a dilated kernel silently produced garbage for windows past the
  first stride positions, on both the eager backend and the native
  compiler (they share the window-reduce core). A dilated kernel axis
  gets an `as_strided` stride > 1, so the sliding-window view aliases
  fewer physical elements than its logical size; MLX's strided-reduce
  fast path then read past the aliased buffer. The view is now
  materialised contiguously before the reduce when any dilation > 1
  (the common non-dilated pooling path is unchanged and stays
  copy-free).
