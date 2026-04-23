### Changed

- Hex consumers now receive a precompiled NIF
  (`libemily.{so,dylib}` + `mlx.metallib`) instead of source. First
  `mix compile` downloads the matching `emily-nif-<v>-<variant>-
  <target>.tar.gz` from the emily GitHub release for the pinned
  version, verifies a SHA256 pinned in `@nif_checksums`, and extracts
  into `priv/`. No cmake / Xcode / C++ toolchain is needed on the
  consumer side.
- In-repo / CI builds now clone MLX's source via a Mix git dep
  (`:mlx_src`) and build libmlx from source; `release-mlx.yml` is
  retired.
- Variant selection is unified under the `:variant` app-config key
  (`:aot` | `:jit`). Contributors flip variants via
  `EMILY_MLX_VARIANT=jit` (read by `config/config.exs`); consumers
  set `config :emily, variant: :jit` in their own
  `config/config.exs`. The old `:mlx_variant` key and
  `config/local.exs` override are gone.

### Added

- `.github/workflows/release-nif.yml` — on bare-semver tag push,
  builds + uploads the precompiled NIF for each `(variant × target)`
  cell and prints a ready-to-paste `@nif_checksums` line in the job
  summary.

### Removed

- `config/local.exs` override (obsoleted by the env-var plumbing).
- `.github/workflows/release-mlx.yml` (MLX build is folded into the
  NIF workflow).
- `scripts/build-mlx-prebuilt.sh` (superseded by in-tree
  `scripts/build-mlx.sh`).
- `scripts/smoke-test-package.sh` and the tagged `smoke-test` job in
  `ci.yml` (simulated a source-compile consumer, no longer
  applicable).

See `MAINTAINING.md` for the updated release flow.
