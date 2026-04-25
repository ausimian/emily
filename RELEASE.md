### Changed

- Hex consumers now receive a precompiled NIF
  (`libemily.{so,dylib}` + `mlx.metallib`) instead of source. First
  `mix compile` downloads the matching `emily-nif-<v>-<variant>-
  <target>.tar.gz` (and its `.sha256` sidecar) from the emily GitHub
  release for the pinned version, verifies the tarball against the
  published SHA256, and extracts into `priv/`. No cmake / Xcode /
  C++ toolchain is needed on the consumer side.
- In-repo / CI builds now clone MLX's source via a Mix git dep
  (`:mlx_src`) and build libmlx from source; `release-mlx.yml` is
  retired.
- Variant selection is unified under the `:variant` app-config key
  (`:aot` | `:jit`). Contributors flip variants via
  `EMILY_MLX_VARIANT=jit` (read by `config/config.exs`); consumers
  set `config :emily, variant: :jit` in their own
  `config/config.exs`. The old `:mlx_variant` key and
  `config/local.exs` override are gone.
- macOS default cache location moves from `~/Library/Caches/emily/`
  to `DARWIN_USER_CACHE_DIR` (`/private/var/folders/<hash>/C/emily`)
  — the per-user sandboxed cache root Apple's own sandboxed apps
  use. Persistent across reboots, lives outside `~/Library/`.
  Linux / Windows still use the XDG convention. Override via
  `EMILY_CACHE`. Existing macOS users can `rm -rf
  ~/Library/Caches/emily/` to reclaim the orphaned data after
  upgrade.
- NIF object files move from the user-level cache to
  `$(MIX_APP_PATH)/obj/` (i.e. `_build/<env>/lib/emily/obj/`). As a
  consequence, plain `mix clean` now correctly removes them via the
  existing Makefile rule — they were previously left behind because
  `make clean` didn't see the cache-dir env vars.

### Added

- `.github/workflows/release-nif.yml` — on bare-semver tag push,
  builds the precompiled NIF for each `(variant × target)` cell and
  uploads tarball + `.sha256` sidecar to a draft GitHub release.
  `workflow_dispatch` is also wired for out-of-band rebuilds
  (artefacts go to workflow storage; the release is untouched).
- `mix clean.mlx` — wipes the MLX install dir(s) under the cache.
  Plain `mix clean` deliberately preserves them since rebuilding
  MLX from source is ~5-7 minutes.

### Fixed

- MLX source builds are now atomic. The build script installs into
  `${PREFIX}.staging` and only `mv`s onto the final path after the
  artefact sanity checks pass; an EXIT trap wipes the scratch dirs
  on failure. Previously, an interrupted build (Ctrl-C, killed
  process, concurrent run) left an empty install dir that
  subsequent `mix compile` runs misread as "MLX is already
  installed", silently skipping the build and bombing out in
  `elixir_make` with `make: *** No rule to make target
  '.../mlx.metallib'`. The compile-time check now requires both
  `lib/libmlx.a` and `lib/mlx.metallib` to be present before
  trusting the dir.
- Concurrent invocations of `build-mlx.sh` against the same install
  prefix are now serialised via a `mkdir`-based lock with
  stale-PID reclaim. ElixirLS uses its own build path
  (`.elixir_ls/build/...`) so an LSP-driven `mix compile` and a CLI
  `mix compile.emily_mlx --force` lock on *different*
  `Mix.Project.with_build_lock` keys and freely raced into the same
  MLX cache dir, clobbering each other's `${PREFIX}.build/`
  mid-build and surfacing as `clang ... Rename failed: ... No such
  file or directory` during Metal-shader compilation.
- CMake's FetchContent sub-build of metal_cpp / json / fmt during
  configure runs with `CMAKE_BUILD_PARALLEL_LEVEL=1`, dodging a
  race in its download → extract → rename → stamp-touch pipeline
  that surfaced as `getcwd: cannot access parent directories`
  followed by `cd: <dir>/_deps: No such file or directory`. The
  main MLX build still runs at full NCPU jobs.
- The MLX scratch build dir (`${PREFIX}.build`) is preserved on
  configure failure so `CMakeError.log` survives for diagnostics.

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
