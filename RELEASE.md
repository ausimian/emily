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
- `Emily.Native` now annotates NIF errors with operation, input
  shape/dtype, options, and worker context. `ArgumentError` and
  `RuntimeError` raised from async ops get an `Emily.Native context:
  op=… inputs=[…] options=[…] stream=…` suffix, so common failures
  (shape mismatches in `matmul`, divisibility errors in `quantize`,
  mask shape bugs in `fast_scaled_dot_product_attention`, etc.) are
  diagnosable from the message alone. The error-formatting path is
  total — bad context maps degrade to `?` markers rather than masking
  the underlying NIF error.
- `Emily.Memory` — public allocator API for long-running serving and
  training workloads that need to observe and manage MLX memory
  without reaching into `Emily.Native`. Exposes `stats/0` (active,
  peak, and cached bytes, also emitting `[:emily, :memory, :stats]`),
  `reset_peak/0`, and `clear_cache/0`. Documented under the README's
  Observability section and grouped with `Emily.Telemetry` in the
  ExDoc sidebar.
- `config :emily, fallback: :silent | :warn | :raise` — strict
  fallback modes for development and CI. `:silent` (the default)
  preserves today's behaviour; `:warn` emits the one-shot
  `Logger.warning` per `{op, input_shapes}` pair previously gated by
  `:warn_on_fallback`; `:raise` raises `RuntimeError` with op,
  shapes, and dtypes on entry, letting CI fail the build when a hot
  path unexpectedly routes through `Nx.BinaryBackend`. An invalid
  `:fallback` value raises `ArgumentError` on the first fallback so
  typos surface immediately.

### Changed

- `Emily.Telemetry.memory_stats/0` now delegates to
  `Emily.Memory.stats/0`. Behaviour is unchanged — same event,
  measurements, and return shape — but new code should prefer the
  `Emily.Memory` entry point.
- The legacy `config :emily, :warn_on_fallback, true` boolean is
  soft-deprecated in favour of `:fallback`. It is still honoured
  when `:fallback` is unset (`true` → `:warn`); when both are set,
  `:fallback` wins.
