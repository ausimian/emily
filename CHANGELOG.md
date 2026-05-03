# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- %% CHANGELOG_ENTRIES %% -->

## 0.3.4 - 2026-05-03

### Fixed

- `Nx.LinAlg.svd(tensor, full_matrices?: false)` on rank-2 inputs no
  longer routes through MLX's full-matrices SVD and post-slices —
  MLX's SVD has no thin switch, so the old path materialised the full
  m × m U on device and instantly OOM'd Metal for tall matrices like
  the Qwen3-0.6B embedder kernel (151936 × 1024 → ~92 GB U). The thin
  case now computes `G = MᵀM → eigh → S, V; U = MV / S` (or the
  symmetric `MMᵀ` route for wide matrices), keeping the decomposition
  at min(m, n)². See the `Emily.Backend` moduledoc Divergences section
  for the numerical caveat (the Gram step squares M's condition
  number). Refs #84.
- `mix docs` runs cleanly. The MNIST notebook referenced
  `Axon.Loop.trainer/2` (no such arity); three other inline references
  resolved to `@doc false` callees in upstream libraries
  (`Nx.Defn.Expr.optional/3`, `Bumblebee.Layers.rms_norm/2`) and
  triggered autolinker warnings on every doc build. The notebook now
  uses the correct `trainer/3` arity, and the prose references have
  been reshaped so the autolinker no longer follows them, keeping the
  build warning-free for future `--warnings-as-errors` enforcement.
  Refs #83.

## 0.3.3 - 2026-05-03

### Fixed

- `Emily.Compiler` now silently drops options it doesn't recognise
  instead of raising `ArgumentError`. This matches the behaviour of
  `Nx.Defn.Evaluator` and EXLA, and restores compatibility with
  higher-level libraries that forward caller-supplied options through
  the JIT compiler — notably `Axon.build/2`, whose contract states
  that "all other options are forwarded to the underlying JIT
  compiler". Hit when running a Bumblebee-built Axon model with
  `Axon.predict(..., global_layer_options: [output_hidden_states:
  true])` under Emily as the global defn compiler. Refs #81.

## 0.3.2 - 2026-04-25

## 0.3.1 - 2026-04-25

### Fixed

- Precompiled NIF download no longer times out on the `:peer.call/4`
  default 5s `gen_server.call` deadline. Consumers installing
  `{:emily, "~> 0.3"}` on a cold cache could see `:gen_server.call`
  timeouts while fetching the multi-MB tarball; the `.sha256` sidecar
  fit in the window but the main asset did not. The peer RPC now runs
  with `:infinity` so httpc's own request timing drives cancellation.

## 0.3.0 - 2026-04-25

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

## 0.2.2 - 2026-04-23

### Fixed

- MLX prebuilt download now runs on a peer VM (`:peer.start_link/1` with
  stdio connection) so it is unaffected by Mix's code-path pruning
  during dep compilation. Previous releases crashed in the tagged
  `smoke-test` CI lane with `{:error, :nofile}` / "module :public_key
  is not available" on clean caches, because Mix removed the
  `:ssl`/`:public_key`/`:asn1`/`:inets` ebin directories from the
  parent VM's code path even though the apps were started. The peer
  node has a fresh code path, so standard `httpc` + `public_key` work
  without further shimming.

## 0.2.1 - 2026-04-22

### Fixed

- **`mix compile` crash on a cold MLX download in a clean consumer
  project.** `http_download!/2` in `mix.exs` called
  `:public_key.cacerts_get/0` right after
  `Application.ensure_all_started(:ssl)`. The app-start path pulled
  `:public_key` in transitively, but the module itself was not
  guaranteed to be loaded at call time — the tag-triggered Hex
  smoke test on CI blew up with
  `UndefinedFunctionError ... module :public_key is not available`
  on 0.2.0. `http_download!` now force-loads the module via
  `:code.ensure_loaded/1` before touching it. Any checkout with a
  populated `~/Library/Caches/emily/mlx-<v>-*` directory skipped
  this path, which is why the break only surfaced in the first
  clean CI run.

## 0.2.0 - 2026-04-22

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
  `defn`-callable — a fallback via `Nx.Defn.Expr`'s `optional/3` would
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

- **MLX prebuilt download replaces the vendored source build.** The
  `vendor/mlx` submodule and the cmake-from-source path are gone.
  `mix compile` now downloads a SHA256-verified `libmlx.a` +
  `mlx.metallib` + headers tarball for the pinned `@mlx_version` from
  this repo's releases into `$EMILY_CACHE` and links the NIF against
  it directly. Consumer prerequisites drop from "Xcode + Metal
  toolchain + cmake + submodule checkout" to just macOS Apple Silicon.
  The JIT / no-JIT switch moves from the `EMILY_MLX_JIT` env var to
  `config :emily, mlx_variant: :jit | :no_jit` in `config/config.exs`
  (default `:no_jit`); variant is read via `Config.Reader.read!` at
  project load, so a gitignored `config/local.exs` is the supported
  per-checkout override. Version bumps are a single-commit change of
  `@mlx_version` + `@mlx_checksums` in `mix.exs`, paired with a new
  `mlx-<version>` GitHub release produced by `release-mlx.yml`. First
  MLX pin under the new scheme: **0.31.2**.
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
- **Parallel NIF C++ build.** `elixir_make` doesn't pass `-j` by
  default and `mix.exs` didn't set `:make_args`, so every `.cpp`
  in `c_src/` compiled serially. `mix.exs` now passes
  `-j#{System.schedulers_online()}` through, and the vestigial
  `JOBS` / `MAKE_JOBS` pair in the `Makefile` (computed but never
  referenced) has been removed. On an 8-core M-series, a clean NIF
  build drops from ~19 s to ~7 s.

## 0.1.2 - 2026-04-19

### Fixed

- **HexDocs source links.** `mix.exs`'s `source_url_pattern`
  prepended a `v` prefix to the version tag, but the project's
  release convention (via `mix publisho`) uses bare semver tags.
  The generated `[source]` links in HexDocs pointed at nonexistent
  `v<version>` tags. Dropped the prefix so links resolve to the
  actual tag.

## 0.1.1 - 2026-04-19

Initial release. See the git history for per-milestone detail.

### Added

- **Nx backend.** `Emily.Backend` implements every required
  `Nx.Backend` callback against MLX, with transparent fallback to
  `Nx.BinaryBackend` for ops without a native primitive.
- **Defn compiler.** `Emily.Compiler` runs `defn` / `Nx.Serving` /
  Bumblebee on Emily; pins the result backend and caps partition
  concurrency so `Nx.Serving` stays compatible.
- **Fused transformer kernels.** `Emily.Fast` exposes
  `mx::fast::rms_norm`, `layer_norm`, `rope`, and scaled-dot-product
  attention as defn-callable helpers with composed-defn fallbacks
  for non-Emily backends. `Emily.Bumblebee.FastKernels` rewrites a
  Bumblebee Axon graph to call the fused kernels in place; declared
  as an optional dep on `:axon` + `:bumblebee`, elides cleanly if
  either is absent.
- **Affine group-wise quantization.** `Emily.QuantizedWeight` and
  `Emily.Quantization` wrap MLX `quantize` / `dequantize` /
  `quantized_matmul` for int2 / int4 / int8 inference.
  `Emily.Quantization.dequantize_defn/1` provides a defn-native
  dequantize for use inside Axon forward passes.
- **Mixed-precision training.** `Emily.MixedPrecision` ships the
  bf16 recipe: `cast_params` for the forward pass, f32 master
  weights, dynamic loss scaling with overflow detection.
- **Per-process Metal streams.** `Emily.Stream` lets each BEAM
  process own its own Metal command queue, enabling concurrent
  inference on a shared model.
- **Zero-copy `to_binary`.** `Nx.to_binary/1` on an Emily tensor
  returns a BEAM resource binary aliasing the MLX buffer — no memcpy.
- **Native gradient + training primitives.** `gather`, `scatter`,
  `scatter_add`, `conv`, and the window-reduction family lower
  directly to MLX so `Nx.Defn.grad` and CNN training stay native.
- **Native linalg.** `lu`, `svd`, `qr`, `cholesky`, `eigh`, `solve`,
  and `triangular_solve` dispatch to `mx::linalg::*` instead of
  rounding through `Nx.BinaryBackend`.
- **Telemetry.** `[:emily, :eval, *]`, `[:emily, :to_binary, *]`,
  `[:emily, :fallback, *]`, and `[:emily, :memory, :stats]` span
  events; opt-in one-shot fallback warnings via
  `config :emily, :warn_on_fallback, true`.
- **Compile-time debug flags.** `:debug_bounds_check` and
  `:debug_detect_nan_inf` re-enable runtime assertions on hot paths;
  default off with zero runtime cost.
- **Bumblebee conformance.** End-to-end suites for DistilBERT,
  Qwen3-0.6B (dense and quantized), ViT-base, and Whisper-tiny,
  pinned against HuggingFace reference values.
- **Worker-thread dispatch.** Each MLX stream is owned by a
  dedicated OS thread. NIFs enqueue work on the worker and return
  immediately; the worker posts the result back to the caller via
  `enif_send`, and the public wrapper awaits it with `receive`. No
  BEAM scheduler (regular or dirty) blocks on MLX work, and the
  per-thread Metal `CommandEncoder` state stays consistent regardless
  of how the BEAM migrates Elixir processes between schedulers.
- **Vendored MLX build.** MLX is built from source via cmake from
  `vendor/mlx` (git submodule); no prebuilt download. Build cache
  keyed on the submodule SHA under `~/Library/Caches/emily/`.
- **Documentation.** Per-module HexDocs, five runnable Livebooks
  (`notebooks/distilbert_qa.livemd`,
  `notebooks/qwen3_quantized.livemd`,
  `notebooks/mnist_training.livemd`,
  `notebooks/whisper_transcription.livemd`,
  `notebooks/fast_kernels.livemd`), and worked Bumblebee examples in
  the conformance suite.
