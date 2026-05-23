### Added

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
