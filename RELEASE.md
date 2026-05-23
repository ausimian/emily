### Added

- `Emily.Memory` — public allocator API for long-running serving and
  training workloads that need to observe and manage MLX memory
  without reaching into `Emily.Native`. Exposes `stats/0` (active,
  peak, and cached bytes, also emitting `[:emily, :memory, :stats]`),
  `reset_peak/0`, and `clear_cache/0`. Documented under the README's
  Observability section and grouped with `Emily.Telemetry` in the
  ExDoc sidebar.

### Changed

- `Emily.Telemetry.memory_stats/0` now delegates to
  `Emily.Memory.stats/0`. Behaviour is unchanged — same event,
  measurements, and return shape — but new code should prefer the
  `Emily.Memory` entry point.
