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
  Unsupported constructs — `while` loops and arbitrary BEAM `reduce` functions —
  raise a clear compile-time error rather than silently falling back, so
  generation loops stay driven from Elixir.

  An opt-in compiled eval mode additionally wraps the replay in
  `mlx::core::compile`, fusing the elementwise runs (rms-norm, softmax, SiLU
  gating, residual adds) the replay leaves as separate kernels — measured at
  ~1.5–1.6× over the plain replay on a decode-shaped transformer block.

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
