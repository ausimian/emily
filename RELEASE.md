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

### Fixed

- `Emily.Fast.rope_with_freqs/4` rotated by the wrong angles on the
  Emily backend: MLX expects periods, not the inverse frequencies the
  function takes. It now matches its documented contract.

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
