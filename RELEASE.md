### Changed

- Upgraded to Nx 0.12 / Bumblebee 0.7 / Axon 0.8. Nx 0.12 replaces
  the optional-callback list (`lu`, `svd`, `qr`, `cholesky`, `eigh`,
  `solve`, `take`, `take_along_axis`, `fft2`, `ifft2`,
  `cumulative_*`, `logical_not`, `all_close`) with a single
  generic `Nx.Backend.block/4` dispatch keyed on `Nx.Block.*`
  structs. `Emily.Backend` now routes every previously-native op
  through `block/4`, preserving the MLX fast paths without losing
  the BinaryBackend fallback when an unknown block arrives. Existing
  `Emily.Backend` consumers see no behavioural change.
- Migrated `Emily.Fast.*` from the now-removed
  `Nx.Defn.Expr.optional/3` extension point to `Nx.block/4`. Each
  fused kernel (`rms_norm`, `layer_norm`, `rope`, `rope_with_freqs`,
  `scaled_dot_product_attention` with and without mask/sinks) now
  emits an `Emily.Fast.Block.*` struct that `Emily.Backend.block/4`
  pattern-matches to the matching `mx::fast::*` NIF. The
  composed-defn fallbacks under non-Emily backends are unchanged.
- Bumblebee 0.7 ships Qwen3 first-class, so
  `notebooks/qwen3_quantized.livemd` no longer needs the `main`-ref
  Bumblebee pin from the 0.6.3 era.

### Added

- `Nx.rfft/2` and `Nx.irfft/2` support. The underlying
  `Native.rfftn` / `Native.irfftn` NIFs were already in place from
  earlier MLX work; Nx 0.12 surfaces these as backend-block ops so
  Emily wires them up at no MLX-side cost.
- Smoke tests for three new Bumblebee 0.7 model families on
  `Emily.Backend`: NomicBERT (`:nomic_embeddings`), SmolLM3
  (`:smollm3`), and ModernBERT (`:modernbert`). All three drive a
  tiny synthetic spec end-to-end through `Axon.predict` so they
  remain offline-friendly; tagged `:conformance`.
- A `[:emily, :block, :fallback]` telemetry event fires whenever
  `Emily.Backend.block/4` falls through to the supplied default
  `fun`. Surfaces ops we used to handle natively but now land on
  the composed-defn path — useful in soak runs to spot silent
  regressions after a Bumblebee bump.

### Removed

- `{:f8_e4m3fn, 8}` (introduced in Nx 0.11) is rejected at the
  backend boundary with the same "no MLX primitive" `ArgumentError`
  pattern as `{:f, 64}`. MLX has no float-8 dtype; cast to `:f16` or
  `:bf16`.
