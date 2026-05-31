### Security

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
