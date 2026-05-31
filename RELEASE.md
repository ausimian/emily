### Security

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
