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
