### Added

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
