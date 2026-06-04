### Added

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
