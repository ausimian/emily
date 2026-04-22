### Added

- **MLX prebuilt-release workflow
  (`.github/workflows/release-mlx.yml`).** Manual workflow that
  builds `libmlx.a` + `mlx.metallib` + headers from a chosen
  `ml-explore/mlx` tag and uploads the tarball to a draft GitHub
  release tagged `mlx-<version>` on this repo. Used to produce the
  prebuilts that Emily's compile step downloads instead of the
  previous source-build path. To cut a new MLX prebuilt release:
  1. Run the workflow with `build_type=no-jit` on macos-14
     (produces `mlx-<v>-macos-arm64-aot.tar.gz`).
  2. Run it again with `build_type=jit` on macos-26 (produces
     `mlx-<v>-macos-arm64-jit.tar.gz`).
  3. Copy the two SHA256s from the draft release's `.sha256`
     sidecars into `@mlx_checksums` in `mix.exs`.
  4. Un-draft the release so consumers can fetch.
  The heavy lifting sits in `scripts/build-mlx-prebuilt.sh`, which
  runs standalone for local debugging:
  `scripts/build-mlx-prebuilt.sh path/to/mlx-src 0.31.2 0`.
- **`Emily.Fast.einsum/2`** — eager-only wrapper around MLX's
  path-optimised `mx::einsum`. Accepts a standard Einstein-summation
  string and a list of `Emily.Backend`-backed tensors; MLX picks the
  contraction order internally. Operands on any other backend raise
  `ArgumentError` with a transfer-first message. The helper is a
  direct-call eager helper (same pattern as
  `Emily.Quantization.quantized_matmul/2`) and is intentionally **not**
  `defn`-callable — a fallback via `Nx.Defn.Expr.optional/3` would
  require a full einsum-string parser and is deferred until a user
  needs cross-backend composability.

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
