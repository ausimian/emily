# Release notes for next release

## Added

- M0 scaffold: mix project, MLX 0.25.1 prebuilt fetch pipeline,
  Makefile wiring `fine` + MLX, `Emily.Native` NIF surface for tensor
  round-trip, application supervisor skeleton, smoke test suite.
- M1 — `Emily.Native` op inventory. Shared headers in `c_src/emily/`
  (dtype mapping, Tensor resource, helpers); per-category op files
  under `c_src/ops/`:
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
    `argmax`/`argmin`, cumulative `cumsum`/`cumprod`/`cummax`/`cummin`/
    `logcumsumexp`.
  - Shape: `reshape`, `transpose`, `squeeze`, `expand_dims`,
    `broadcast_to`, `concatenate`, `stack`, `flatten`, `tile`,
    `swapaxes`, `pad`, `repeat`.
  - Sort family: `sort`, `argsort`, `partition`, `argpartition`,
    `topk`.
  - Indexing: `slice`, `take`, `where`, `take_along_axis`,
    `put_along_axis`, `scatter_add_axis`.
  - Misc: `clip`, `roll`, `softmax`, `array_equal`.
  - Linalg: `matmul`, `tensordot`, `outer`, `inner`.
  - Convolution: `conv_general` (N-D with asymmetric padding,
    dilation, groups, flip).
  - Random: `random_key`, `random_split`, `random_uniform`,
    `random_normal`, `random_randint`, `random_bernoulli`,
    `random_gumbel`, `random_categorical` — keys passed as optional
    tensor args (nil uses MLX's default key sequence).
  - FFT: `fftn`, `ifftn`, `rfftn`, `irfftn`.
  - Memory: `get_active_memory`, `get_peak_memory`,
    `reset_peak_memory`, `get_cache_memory`, `clear_cache` — exposed
    so the soak harness can observe allocator state.
- `Emily.Native.to_binary/1` routes through `mx::contiguous` so
  strided views (transpose/slice/swapaxes/broadcast) materialize
  correctly.
- `test/support/tensor_helpers.ex` — shared build/inspect helpers.
- `test/soak/memory_test.exs` (`@tag :soak`, excluded by default) —
  5000-iteration allocate/eval/drop loop; asserts MLX active memory
  returns within 1 MB of baseline after `clear_cache`.
- `test/emily/dtype_matrix_test.exs` — smoke matrix covering every
  supported dtype across creation, cast, unary (float + numeric),
  binary, reductions, and comparisons.
- Makefile compiles `c_src/**/*.cpp` recursively.

## Notes

- Ops files use anonymous namespaces to prevent NIF function names
  (`sin`, `log1p`, `sqrt`, ...) from colliding with C math-library
  symbols pulled in by MLX headers.
- Deferred beyond M1: the full `scatter`/`scatter_add`/... family with
  vector-of-indices (only the axis-aligned forms are bound),
  `hadamard_transform`, quantized matmul, `linalg.*` decompositions
  (LU, QR, Cholesky, SVD). These will be added opportunistically when
  M2/M3 callers need them.
