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
