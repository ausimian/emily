### Added

- `Emily.QuantizedWeight.from_dense/2` accepts `bits: 1` for `"affine"`
  quantization, which MLX 0.32.4 added. Each element dequantizes to its
  group's minimum or maximum, packed 32 to a `u32`. 1-bit weights work
  with `Emily.Quantization.quantized_matmul/2`,
  `quantized_matmul_defn/2` and `dequantize_defn/1`, and
  `Emily.Quantization.defn_supported_bits/0` now includes `1`. On the
  GPU, MLX widens every group's range to include zero, so a group whose
  values are all negative dequantizes to its minimum or `0`.

### Changed

- Updated the pinned MLX to 0.32.4. Emily picks up its upstream fixes
  without API changes. They cover strided and sliced views, reductions
  over very large arrays, FFTs of size one, empty unsigned sums and
  products on Metal, and a deadlock during eval cleanup. Split-K
  quantized matmul now keeps its partial sums in float32. The release
  also speeds up quantized matvec when the output size isn't a multiple
  of 8, GEMMs with a thin N, and vector SDPA decode, and uses less
  memory for SDPA with head dim 512 on chips before M5.
