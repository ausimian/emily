### Changed

- Updated the pinned MLX to 0.32.3. This is a maintenance bump that picks
  up upstream correctness fixes (NaN propagation in arg reductions,
  quantized matmul with a quantized dim that is not a multiple of 32,
  scans and sorts over zero-size axes, quantizing sliced arrays) and
  faster GQA decode attention, all transparently. No API changes.
