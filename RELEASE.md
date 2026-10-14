### Changed

- Updated the pinned MLX to 0.32.4. Emily picks up its upstream fixes
  without API changes. They cover strided and sliced views, reductions
  over very large arrays, FFTs of size one, empty unsigned sums and
  products on Metal, and a deadlock during eval cleanup. Split-K
  quantized matmul now keeps its partial sums in float32. The release
  also speeds up quantized matvec when the output size isn't a multiple
  of 8, GEMMs with a thin N, and vector SDPA decode, and uses less
  memory for SDPA with head dim 512 on chips before M5.
