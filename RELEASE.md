### Added

- `Emily.Fast.scaled_dot_product_attention/4` and
  `scaled_dot_product_attention_with_mask/5` accept `force_fused: true`.
  MLX then always uses one of its fused attention kernels instead of
  choosing per shape, and raises `ArgumentError` if no fused kernel
  supports the shape. This keeps attention memory bounded for
  long-context prefill with head dims 192 and 256 (for example Gemma),
  where MLX otherwise builds the full attention-score matrix, at some
  cost in prefill throughput. Off by default; it works in both eager
  and natively compiled code.
- `Emily.Fast.rope/3` and `rope_with_freqs/4` accept a `{batch}` offset
  as well as a scalar, so each entry along the leading axis starts at its
  own position, for example in batched decode.
- `Emily.Bumblebee.FastKernels.apply/2` accepts `force_fused_sdpa: true`,
  which passes `force_fused: true` to every rewritten attention layer.

### Fixed

- `Emily.Bumblebee.FastKernels.apply/1` now rewrites the attention and
  RoPE layers of Bumblebee models. Before, it only rewrote RMSNorm and
  LayerNorm: it never recognised Bumblebee's attention and RoPE layers,
  so the fused SDPA and RoPE kernels never ran. Decoder models such as
  Qwen3 now get the fused kernels, whose advantage grows with context
  length. The newly active paths are checked against stock Bumblebee,
  including the `:linear`, `:dynamic`, `:longrope` and `:llama3` RoPE
  scaling strategies, padded and batched position ids, and causal decode
  against a KV cache.
- `Emily.Fast.rope_with_freqs/4` rotated by the wrong angles on the
  Emily backend: MLX expects periods, not the inverse frequencies the
  function takes. It now matches its documented contract.
- The defn fallback of `Emily.Fast.scaled_dot_product_attention/4` (used
  on non-Emily backends) applied an inverted causal mask, letting each
  query attend to later keys instead of earlier ones.
- The defn fallback of `Emily.Fast.rope/3` and `rope_with_freqs/4` (used
  on non-Emily backends) added the offset after applying `:scale`; it now
  scales the whole position, as MLX does.

### Changed

- Updated the pinned MLX to 0.32.3. This is a maintenance bump that picks
  up upstream correctness fixes (NaN propagation in arg reductions,
  quantized matmul with a quantized dim that is not a multiple of 32,
  scans and sorts over zero-size axes, quantizing sliced arrays) and
  faster GQA decode attention, all transparently. No API changes.
- Building MLX from source (a git checkout of Emily) now needs Xcode 16
  or later. MLX 0.32.1 started using a Metal standard library
  type that Xcode 15 does not ship. The precompiled NIFs are unaffected
  and still support macOS 14.
