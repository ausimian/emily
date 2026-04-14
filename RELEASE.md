# Release notes for next release

## Added

- M0 scaffold: mix project, MLX 0.25.1 prebuilt fetch pipeline,
  Makefile wiring `fine` + MLX, `Emily.Native` NIF surface for tensor
  round-trip, application supervisor skeleton, smoke test suite.
- M1 (partial) — `Emily.Native` op inventory. Shared headers in
  `c_src/emily/` (dtype mapping, Tensor resource, helpers);
  per-category op files under `c_src/ops/`:
  - Creation: `zeros`, `ones`, `full`, `arange`, `eye`.
  - Cast: `astype`.
  - Unary elementwise: `negative`, `abs`, `sign`, `floor`, `ceil`,
    `sqrt`, `rsqrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`,
    trig/inverse-trig/hyperbolic family, `sigmoid`, `erf`, `erfinv`,
    `square`, `reciprocal`, `logical_not`, `bitwise_invert`, `isnan`,
    `isinf`, `isfinite`, `conjugate`, `real`, `imag`, `stop_gradient`,
    `round` (with decimals).
  - Binary elementwise: `add`, `subtract`, `multiply`, `divide`,
    `floor_divide`, `remainder`, `power`, `maximum`, `minimum`,
    `logaddexp`, `arctan2`.
  - Compare: `equal`, `not_equal`, `less`, `less_equal`, `greater`,
    `greater_equal`.
  - Logical: `logical_and`, `logical_or`.
  - Bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`,
    `right_shift`.
  - Reductions (axes + keepdims): `sum`, `mean`, `prod`, `max`, `min`,
    `all`, `any`, `logsumexp`; plus `var`/`std` with `ddof`,
    `argmax`/`argmin`, cumulative `cumsum`/`cumprod`/`cummax`/`cummin`.
  - Shape: `reshape`, `transpose`, `squeeze`, `expand_dims`,
    `broadcast_to`, `concatenate`, `stack`, `flatten`, `tile`,
    `swapaxes`, `pad`, `repeat`.
  - Indexing: `slice`, `take`, `where`.
  - Linalg: `matmul`, `tensordot`, `outer`, `inner`.
- `Emily.Native.to_binary/1` now routes through `mx::contiguous` so
  strided views (transpose/slice/swapaxes/broadcast) materialize
  correctly.
- Makefile compiles `c_src/**/*.cpp` recursively.

## Notes

- Ops files use anonymous namespaces to prevent NIF function names
  (`sin`, `log1p`, `sqrt`, ...) from colliding with C math-library
  symbols pulled in by MLX headers.
- Deferred to later iterations of M1: sort/argsort, clip,
  `slice_update`, `take_along_axis`, `scatter*`, convolutions,
  `hadamard_transform`, random ops, FFT, quantized ops,
  memory-stats/soak tests. Tracked for M1 completion before moving to
  M2 (Backend).
