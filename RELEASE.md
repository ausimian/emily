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
