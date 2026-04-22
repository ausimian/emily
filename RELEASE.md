### Fixed

- **`Nx.top_k/2` on Emily tensors.** The backend's `top_k/3`
  override pattern-matched `out` as a single `%Nx.Tensor{}` and
  returned a single tensor, but the real Nx callback contract takes
  `{out_values, out_indices}` and returns a `{values, indices}`
  tuple. Any call to `Nx.top_k` raised `FunctionClauseError`.
  Dropped the override so Nx falls back to `argsort(:desc) +
  take_along_axis + slice_along_axis`, each of which routes
  through Emily's backend.

### Changed

- **Microscaled quantization modes on `Emily.QuantizedWeight`.** The
  container now carries a `:mode` field (default `"affine"`) and
  accepts `"mxfp4"`, `"mxfp8"`, `"nvfp4"` — MLX's full
  `QuantizationMode` enum (`vendor/mlx/mlx/primitives.h:155`).
  `from_dense/2`, `to_dense/1`, and `Emily.Quantization.quantized_matmul/2`
  all thread the mode through to MLX; mode-specific
  `{group_size, bits}` constraints are validated up front with a
  clear Emily error before the NIF call. Microscaled modes carry
  a placeholder biases tensor — MLX's `fp_quantize` returns only
  `(wq, scales)`, and the Native layer substitutes `nil` before
  the MLX call. `Emily.Quantization.dequantize_defn/1` is
  affine-only (it's a hand-rolled nibble unpacker) and now raises
  `ArgumentError` on non-affine modes, pointing users at
  `to_dense/1`. Smoke-tested end-to-end on Metal for all four modes
  (Apple Silicon, macOS 26).
- **SDPA attention sinks (`mx::fast::scaled_dot_product_attention`
  `sinks` param).** `Emily.Fast.scaled_dot_product_attention/4` and
  `scaled_dot_product_attention_with_mask/5` now accept an optional
  `:sinks` keyword opt — a per-head tensor broadcastable to
  `{1, heads, 1, 1}` whose entries participate in the softmax
  denominator as extra "null destinations" (StreamingLLM). When
  absent the helpers emit the pre-existing optional-node, so
  `Emily.Bumblebee.FastKernels` and direct callers stay source- and
  bit-compatible. The defn fallback implements the same semantics
  in numerically-stable form; equivalence vs. the fused kernel was
  measured at ~2e-7 max-abs-diff on f32.
- **MLX JIT build no longer patches vendored MLX.** The
  `patches/mlx-jit-nax-gate.patch` workaround (and the
  `maybe_apply_mlx_patches` plumbing in `mix.exs`) has been removed.
  The JIT build now requires the macOS 26.2+ SDK directly, which
  ships `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`;
  the AOT (default) build is unchanged and still works on older
  macOS. Upstream discussion:
  [ml-explore/mlx#3426](https://github.com/ml-explore/mlx/pull/3426).
- **CI matrix split across macOS versions.** The `jit=0` row stays
  on `macos-14` to keep AOT coverage on older macOS; the `jit=1`
  row now runs on `macos-26` so the Metal Performance Primitives
  SDK is available natively.
- **Native axis reversal via `mx::slice` with stride -1.** The
  descending branches of `Nx.sort` and `Nx.argsort` (and
  `Nx.reverse`) previously built an `arange` index tensor and
  gathered with `take`. They now call a new `Native.flip/3` NIF
  that lowers to a single strided slice, saving the index
  allocation and gather kernel per call.
