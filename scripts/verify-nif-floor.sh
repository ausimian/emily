#!/usr/bin/env bash
# Assert a built NIF's macOS floor matches the pinned deployment target, and
# that it imports no libSystem symbols newer than that floor. Run by CI after
# `mix compile` (and usable locally) so the shipped precompiled NIF's minimum
# macOS is a checked invariant rather than an accident of the build host.
#
# Guards specifically against the failure mode that motivated the pin: an aot
# build done on a macOS-26 host silently linking the 26.0 `_Float16` helpers
# `__fmaxf16`/`__fminf16` (hard, non-weak imports from libSystem), which would
# fail to dyld-load on the older macOS the aot variant is supposed to support.
#
# Usage: scripts/verify-nif-floor.sh <libemily.so> <expected-min>
#   e.g. scripts/verify-nif-floor.sh _build/test/lib/emily/priv/libemily.so 14.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <libemily.so> <expected-min>" >&2
  exit 2
fi

SO="$1"
EXPECTED="$2"

[[ -f "$SO" ]] || { echo "error: NIF not found: $SO" >&2; exit 2; }

# 1) Declared minimum — LC_BUILD_VERSION's `minos`.
minos=$(otool -l "$SO" | awk '/LC_BUILD_VERSION/{f=1} f && /^ *minos/{print $2; exit}')
if [[ "$minos" != "$EXPECTED" ]]; then
  echo "error: $SO declares minos '${minos:-<none>}', expected '$EXPECTED'" >&2
  exit 1
fi
echo "ok: $SO declares minos $EXPECTED"

# 2) No above-floor libSystem symbols. macOS 26 added the `_Float16` helpers
#    __fmaxf16/__fminf16; a sub-26 floor must not import them (they resolve
#    from libSystem at load and are absent on earlier macOS).
major=${EXPECTED%%.*}
if (( major < 26 )); then
  if nm -u "$SO" 2>/dev/null | grep -qE '___(fmaxf16|fminf16)'; then
    echo "error: $SO (floor $EXPECTED) imports macOS-26 _Float16 symbols:" >&2
    nm -u "$SO" | grep -E '___(fmaxf16|fminf16)' >&2
    exit 1
  fi
  echo "ok: $SO imports no macOS-26 _Float16 symbols"
fi

echo "NIF floor verified: $SO -> macOS $EXPECTED"
