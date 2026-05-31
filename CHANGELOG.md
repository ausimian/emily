# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- %% CHANGELOG_ENTRIES %% -->

## 0.6.1 - 2026-05-31

### Changed

- Documentation updated for the 0.6.x release: the README installation
  instructions and the example notebooks now reference
  `{:emily, "~> 0.6"}`.

## 0.6.0 - 2026-05-31

This release is a security-hardening pass over the native (NIF) boundary
and the build/release pipeline: direct `Emily.Native` calls now validate
their arguments instead of trusting Elixir-side normalization,
precompiled-NIF downloads verify against a checksum pinned in the hex
package (a trust root independent of the GitHub release), and the
per-stream worker is bounded and tears down without blocking a BEAM
scheduler. It is backward compatible, but two behaviour changes matter
for high-concurrency callers: the per-worker async queue is now bounded
(`worker_queue_limit`, default 8192) and rejects when full, and a stopped
or dropped worker replies `{:error, :stopped}` to queued callers instead
of running their work.

### Added

- `Emily.Stream.close/1` stops a stream's worker thread deterministically
  instead of waiting for garbage collection: queued operations are
  cancelled (their callers get a `RuntimeError`), the in-flight op
  finishes, and the OS thread is joined off the BEAM schedulers.
- `config :emily, worker_queue_limit: N` (default `8192`) bounds the
  per-worker async queue, and `config :emily, await_timeout: ms` (default
  `:infinity`) sets an optional timeout for awaiting native results.

### Security

- Worker-thread teardown no longer blocks a BEAM scheduler. The resource
  destructor previously drained the worker's entire queue and joined the
  OS thread inline, so collecting a busy stream during GC could stall a
  scheduler. Workers are now joined off-scheduler by a dedicated reaper
  (itself joined at NIF unload), and on stop the worker cancels its
  queued tasks — replying `{:error, :stopped}` — instead of running them.

- The async NIF worker queue is now bounded (`worker_queue_limit`, reject
  when full) so a flood of operations can't grow it without limit and pin
  host/GPU memory, and a stopped or dropped worker now replies
  `{:error, :stopped}` to every queued caller instead of leaving it
  blocked forever. `Emily.Native.worker_queue_depth/1` exposes the depth
  for observability.

- The dev/CI source-build path now refuses to trust an MLX install
  directory it doesn't own and keeps the build cache `0700`, so a shared
  or attacker-controlled `EMILY_CACHE` can't plant a `libmlx.a` that is
  then statically linked into the NIF. Fixed system tools (`getconf`,
  `id`, `sw_vers`, plus `xcrun`/`sysctl`/`ps` in `build-mlx.sh`) resolve
  from absolute/system paths rather than `$PATH`, and the MLX-build lock
  records the holder's process start time so a recycled PID can't be
  mistaken for the original holder. Build-time only; no runtime change.

- Precompiled NIF downloads are now verified against checksums pinned
  inside the hex package (`native_checksums.txt`) rather than a `.sha256`
  sidecar fetched from the same GitHub release as the tarball. Because
  the package contents are covered by Hex's package hash in the
  consumer's `mix.lock`, the trust root no longer lives in the mutable
  release. The tarball is also extracted with `:erl_tar` against a strict
  entry allowlist (`libemily.{so,dylib}` + `mlx.metallib`), rejecting
  symlinks, hardlinks, `..` traversal, absolute paths, and unexpected
  entries — closing a path-traversal/arbitrary-write vector in the old
  `tar -xzf` extraction. New `mix emily.checksums` task regenerates the
  pinned file per release.

- Integer arguments crossing the NIF boundary are now range-checked
  before being narrowed from Elixir's `int64` to C++ `int`. Previously an
  out-of-range axis, count, or shape entry wrapped silently (e.g. an axis
  of `2^32 + 3` became `3`), dispatching the wrong MLX operation; and
  unbounded sample counts in `random_split`/`random_categorical` could
  drive huge allocations. Out-of-range values, and negative counts, now
  raise `ArgumentError`. Centralized as `checked_int` / `require_count`
  helpers applied across the reduce, shape, sort, random, index, linalg,
  conv, and fast NIFs.

- Native indexing and window NIFs now validate their vector arguments
  against the tensor rank before indexing, and reject non-positive
  strides, dilations, and window dimensions. Previously a direct
  `Emily.Native` call with a malformed `slice_update` start, a short
  pad/window vector, or a zero window stride could read a C++ vector out
  of bounds or trigger an integer divide-by-zero (SIGFPE) — both of which
  crash the whole BEAM VM rather than raising in the caller. They now
  raise `ArgumentError`.

- `Emily.Native.from_binary/3` now validates tensor shapes at the NIF
  boundary. Dimensions above `INT32_MAX` are rejected (previously they
  silently truncated through MLX's `int32` `ShapeElem`), and the element
  and byte counts are computed with overflow checking. Without this an
  attacker-chosen shape whose element product wrapped (e.g.
  `[2^21, 2^21, 2^22]` → `0`) could pass the binary-size check against an
  undersized — even empty — binary and build an array whose shape outran
  its allocation, an out-of-bounds read on the next `eval`/`to_binary`.

- `Emily.Native.conv_general/8` now rejects a non-positive `groups`
  argument with `ArgumentError` instead of crashing the BEAM VM. MLX's
  convolution checks compute `in_channels % groups`, so `groups <= 0`
  (or a large value that narrows to zero through the `int64 → int`
  conversion) was an integer modulo-by-zero — a SIGFPE that bypassed the
  NIF's exception path and terminated the entire node. The guard
  validates the un-narrowed value at the NIF boundary.

## 0.5.1 - 2026-05-23

### Fixed

- `CHANGELOG.md` — corrected the 0.5.0 entry. The published release
  carried two `### Changed` headings and listed three new-functionality
  items (`mix emily.doctor`, `config :emily, fallback:`, and the
  `Emily.Memory` public allocator API) under Changed rather than
  Added. Merged the duplicate Changed sections, moved the
  new-functionality items to Added, and put items into reverse
  chronological order. No code change.

## 0.5.0 - 2026-05-23

### Added

- `Emily.Quantization.dequantize_defn/1` now supports the `nvfp4`
  microscaled mode in addition to `affine`, `mxfp4`, and `mxfp8` —
  the full MLX `QuantizationMode` enum now runs through the
  defn-native dequant path. `nvfp4` reuses the FP4-E2M1 lane LUT
  from `mxfp4` and the FP8-E4M3 LUT from `mxfp8` (consumed against
  the per-group scale bytes rather than lane codes — the NVIDIA
  microscaled convention uses finer-grained group_size=16 with
  FP8-E4M3 scales instead of mxfp4/mxfp8's group_size=32 with
  FP8-E8M0 scales). Output dtype is bf16 to match
  `QuantizedWeight.to_dense/1`, round-trip is bit-identical (max
  abs diff = 0.0). `Emily.Quantization.Transform` accepts
  `mode: "nvfp4"`.

- `Emily.Quantization.dequantize_defn/1` now supports the `mxfp8`
  microscaled mode in addition to `affine` and `mxfp4`. Each 8-bit
  lane code decodes through a 256-entry FP8-E4M3 lookup table
  precomputed via MLX's `FromFP8` bit-trick (strip sign, shift the
  low 7 bits left by 7 to align the E4M3 exponent into f16's
  exponent field, multiply by 256 for the bias difference, restore
  sign). Per-group scales reuse the FP8-E8M0 decode from the mxfp4
  path. Output dtype is bf16 to match `QuantizedWeight.to_dense/1`,
  and the round-trip is bit-identical (max abs diff = 0.0) on
  realistic data. `Emily.Quantization.Transform` accepts
  `mode: "mxfp8"`; only `nvfp4` (which uses an FP8-E4M3 per-group
  scale instead of FP8-E8M0) remains defn-unsupported.

- `Emily.Quantization.dequantize_defn/1` now supports the `mxfp4`
  microscaled mode in addition to `affine`. Each 4-bit lane code
  decodes through MLX's FP4-E2M1 lookup table (`+0.0, +0.5, +1.0,
  +1.5, +2.0, +3.0, +4.0, +6.0` and their negatives); each u8 scale
  byte decodes through `2^(s - 127)` (FP8-E8M0). Output dtype is
  bf16 to match `QuantizedWeight.to_dense/1`, and the round-trip is
  bit-identical (max abs diff = 0.0) on realistic scale bytes
  because every FP4 LUT entry and every E8M0 power-of-two is exact
  in bf16. `Emily.Quantization.Transform` gains a `:mode` option
  (default `"affine"`, accepts `"mxfp4"`); `mxfp8` and `nvfp4` are
  still defn-unsupported and route through the Native NIF.

- `Emily.Quantization.dequantize_defn/1` now supports int3 and int6
  weights in addition to int2/int4/int8. The new path reads each
  lane's two adjacent u32 words as a u64, shifts by the in-word bit
  offset, and masks — handling the cross-u32 packing MLX uses for
  bit widths that don't divide 32 cleanly. `defn_supported_bits/0`
  now returns `[2, 3, 4, 6, 8]`; quantized Axon graphs rewritten
  via `Emily.Quantization.Transform` (and `Emily.Quantization.Layers.quantized_dense/4`)
  pick the expanded set up automatically. Previously the defn path
  rejected `bits ∈ {3, 6}` and callers had to fall back to
  `QuantizedWeight.to_dense/1` (the Native NIF).

- `ARCHITECTURE.md` — current shape of the library extracted from
  `PLAN.md`. Covers the four-layer dispatch model, the worker-thread
  + per-process-stream concurrency model, the public `Emily.Memory`
  allocator API, the telemetry event catalogue, the
  `:debug_bounds_check` / `:debug_detect_nan_inf` compile-time flags,
  build/packaging notes, the per-layer testing oracle table, and the
  active risk register. Linked from the README under a new
  Documentation section and grouped under "Project" in the HexDocs
  sidebar.
- `ROADMAP.md` — active and future work, separated from the
  historical milestone log. Lists deferred-to-post-1.0 items
  (typed exceptions, GPU interop pointers, source-build doctor
  probes) and the open in-roadmap MLX capability gaps (sparse / MoE
  matmuls, FP8 dtype, `ThreadLocalStream`).
- `mix emily.doctor` — diagnostic Mix task that verifies the local
  Emily runtime installation. Checks the host platform (OS, arch,
  macOS version against the active variant's minimum), the active
  MLX variant, `priv/libemily.so` and `priv/mlx.metallib`, NIF
  loadability, and a tiny `Emily.Backend` smoke test that asserts
  the result didn't silently fall back to `Nx.BinaryBackend`. Checks
  short-circuit: when a prerequisite fails, dependent checks report
  `[skip]` rather than producing cascading noise. Supports
  `--variant aot|jit` for "would this host satisfy :jit?" probes and
  `--help` for usage.
- `config :emily, fallback: :silent | :warn | :raise` — strict
  fallback modes for development and CI. `:silent` (the default)
  preserves today's behaviour; `:warn` emits the one-shot
  `Logger.warning` per `{op, input_shapes}` pair previously gated by
  `:warn_on_fallback`; `:raise` raises `RuntimeError` with op,
  shapes, and dtypes on entry, letting CI fail the build when a hot
  path unexpectedly routes through `Nx.BinaryBackend`. An invalid
  `:fallback` value raises `ArgumentError` on the first fallback so
  typos surface immediately.
- `Emily.Memory` — public allocator API for long-running serving and
  training workloads that need to observe and manage MLX memory
  without reaching into `Emily.Native`. Exposes `stats/0` (active,
  peak, and cached bytes, also emitting `[:emily, :memory, :stats]`),
  `reset_peak/0`, and `clear_cache/0`. Documented under the README's
  Observability section and grouped with `Emily.Telemetry` in the
  ExDoc sidebar.

### Changed

- `PLAN.md` slimmed to its milestone-history role. The current-shape
  sections (architecture diagram, core design decisions, testing
  philosophy, risks-and-mitigations) moved to `ARCHITECTURE.md`;
  goals, non-goals, and deferred-milestone summaries moved to
  `ROADMAP.md`. The M0–M27 milestone narratives, the ratified
  project decisions, and the 2026-04-22 MLX capability audit stay in
  `PLAN.md` as the historical record. The stale "narrow
  `with_stream/2` + `new/1` + `synchronize/1` surface" reference (no
  `synchronize/1` ever shipped) and the planned `set_default_stream/1`
  primary deliverable (removed during the post-M14 fixes) drop out
  with the prologue rewrite.
- `Emily.Native` now annotates NIF errors with operation, input
  shape/dtype, options, and worker context. `ArgumentError` and
  `RuntimeError` raised from async ops get an `Emily.Native context:
  op=… inputs=[…] options=[…] stream=…` suffix, so common failures
  (shape mismatches in `matmul`, divisibility errors in `quantize`,
  mask shape bugs in `fast_scaled_dot_product_attention`, etc.) are
  diagnosable from the message alone. The error-formatting path is
  total — bad context maps degrade to `?` markers rather than masking
  the underlying NIF error.
- The legacy `config :emily, :warn_on_fallback, true` boolean is
  soft-deprecated in favour of `:fallback`. It is still honoured
  when `:fallback` is unset (`true` → `:warn`); when both are set,
  `:fallback` wins.
- `Emily.Telemetry.memory_stats/0` now delegates to
  `Emily.Memory.stats/0`. Behaviour is unchanged — same event,
  measurements, and return shape — but new code should prefer the
  `Emily.Memory` entry point.

## 0.4.0 - 2026-05-17

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
- Runnable Livebooks for each of the three new Bumblebee 0.7
  families: `notebooks/nomic_embeddings.livemd` (NomicBERT
  embeddings with cosine similarity), `notebooks/smollm3_chat.livemd`
  (SmolLM3-3B chat completion with a `<think>` toggle for hybrid
  reasoning), and `notebooks/modernbert_classification.livemd`
  (ModernBERT NLI fine-tune). All three are published under the
  HexDocs Notebooks group.
- A `[:emily, :block, :fallback]` telemetry event fires whenever
  `Emily.Backend.block/4` falls through to the supplied default
  `fun`. Surfaces ops we used to handle natively but now land on
  the composed-defn path — useful in soak runs to spot silent
  regressions after a Bumblebee bump.

### Fixed

- `mix docs` no longer emits autolinker warnings for the
  `Emily.Backend.block/4` and `Nx.Defn.Expr.optional/3` references
  in the `Emily.Fast` and `Emily.Fast.Block` moduledocs. The
  references resolved to `@doc false` callees (the backend callback
  is hidden by `Nx.Backend`, and `optional/3` was removed in Nx 0.12);
  the prose stays, the `Mod.fun/arity` shape is broken up so the
  autolinker no longer follows it. Same pattern as the earlier
  fix in `ee32c7c`.

### Removed

- `{:f8_e4m3fn, 8}` (introduced in Nx 0.11) is rejected at the
  backend boundary with the same "no MLX primitive" `ArgumentError`
  pattern as `{:f, 64}`. MLX has no float-8 dtype; cast to `:f16` or
  `:bf16`.

## 0.3.5 - 2026-05-03

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
  `Axon.Loop`'s `trainer/2` (no such arity); three other inline
  references resolved to `@doc false` callees in upstream libraries
  (`Nx.Defn.Expr`'s `optional/3`, Bumblebee's `rms_norm/2`)
  and triggered autolinker warnings on every doc build. The notebook
  now uses the correct `trainer/3` arity, and the prose references
  have been reshaped so the autolinker no longer follows them,
  keeping the build warning-free for future `--warnings-as-errors`
  enforcement. Refs #83.

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
