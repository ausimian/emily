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
