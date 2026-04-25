#!/usr/bin/env bash
# Build MLX (static libmlx.a + mlx.metallib + headers) from source and
# install into <install-prefix>. Invoked by `mix.exs`'s
# `compile.emily_mlx` compiler step; not intended for direct use.
#
# Usage:
#   scripts/build-mlx.sh <mlx-src-dir> <mlx-version> <jit 0|1> <install-prefix>
#
# <install-prefix> ends up with an {include,lib} layout that mix.exs
# exports to the NIF build via `MLX_INCLUDE_DIR` / `MLX_LIB_DIR`.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <mlx-src-dir> <mlx-version> <jit 0|1> <install-prefix>" >&2
  exit 2
fi

MLX_SRC_DIR="$1"
VERSION="$2"
JIT="$3"
PREFIX="$4"

case "$JIT" in
  0) VARIANT="aot"; METAL_JIT="OFF" ;;
  1) VARIANT="jit"; METAL_JIT="ON"  ;;
  *) echo "error: jit must be 0 or 1 (got: $JIT)" >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: MLX build is macOS-only (uname -s = $(uname -s))" >&2
  exit 2
fi

if [[ ! -d "$MLX_SRC_DIR" ]]; then
  echo "error: MLX source directory not found: $MLX_SRC_DIR" >&2
  echo "       run \`mix deps.get\` to clone the :mlx_src dep" >&2
  exit 2
fi

# Resolve Metal toolchain. CommandLineTools alone can't run `xcrun -sdk
# macosx metal`; if the default developer dir lacks it, fall back to
# Xcode.app. Mirrors the logic the old build-mlx-prebuilt.sh used.
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    echo "==> Using Xcode.app for Metal toolchain (DEVELOPER_DIR=$DEVELOPER_DIR)"
  else
    cat >&2 <<'EOF'
error: Metal toolchain not found. MLX requires the Metal compiler.

Install Xcode from the App Store and run:
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

Or, if Xcode is installed but the Metal Toolchain component is missing:
  xcodebuild -downloadComponent MetalToolchain
EOF
    exit 1
  fi
fi

# Serialise concurrent script invocations against the same install
# prefix. Mix.Project.with_build_lock can't help here: ElixirLS uses
# its own build path (.elixir_ls/build/...) so an LSP-driven
# `mix compile` and a CLI `mix compile.emily_mlx --force` lock on
# *different* keys and freely race into the same MLX cache dir. Both
# invocations would then rm each other's `${PREFIX}.build/` mid-build,
# surfacing as `clang ... Rename failed: ... No such file or
# directory` during Metal-shader compilation.
#
# `flock(1)` isn't shipped on macOS, so we use atomic `mkdir` as the
# lock primitive. The lock dir is keyed on PREFIX, which both
# contexts share. A PID file inside lets us reclaim a stale lock if
# the previous holder died without cleanup.
BUILD_DIR="${PREFIX}.build"
STAGING="${PREFIX}.staging"
LOCK_DIR="${PREFIX}.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"

mkdir -p "$(dirname "$LOCK_DIR")"

acquired_lock=0
printed_wait=0

# Define cleanup + install the trap *before* the lock-acquisition loop
# so that a concurrent-winner short-circuit (or any other early exit)
# still releases LOCK_DIR. STAGING is a half-baked install — always
# wipe. BUILD_DIR holds CMakeFiles/ and CMakeError.log on failure;
# keep it on non-zero exit so diagnostics survive.
cleanup() {
  local exit_code=$?
  rm -rf "$STAGING"
  if (( acquired_lock == 1 )); then
    rm -rf "$LOCK_DIR"
  fi
  if [[ $exit_code -eq 0 ]]; then
    rm -rf "$BUILD_DIR"
  else
    echo "==> Build failed (exit ${exit_code}); preserving ${BUILD_DIR} for diagnostics" >&2
  fi
}
trap cleanup EXIT

while :; do
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_PID_FILE"
    acquired_lock=1
    break
  fi

  holder=""
  [[ -r "$LOCK_PID_FILE" ]] && holder=$(cat "$LOCK_PID_FILE" 2>/dev/null || true)

  if [[ -n "$holder" ]] && ! kill -0 "$holder" 2>/dev/null; then
    echo "==> Reclaiming stale MLX-build lock (dead PID $holder)" >&2
    rm -rf "$LOCK_DIR"
    continue
  fi

  if (( printed_wait == 0 )); then
    echo "==> Waiting for concurrent MLX build${holder:+ (PID $holder)} on ${PREFIX}" >&2
    printed_wait=1
  fi
  sleep 1
done

# A concurrent winner may have completed the install while we waited
# for the lock — re-check and short-circuit if so.
if [[ -f "${PREFIX}/lib/libmlx.a" && -f "${PREFIX}/lib/mlx.metallib" ]]; then
  echo "==> MLX already installed at ${PREFIX} (concurrent build won)"
  exit 0
fi

rm -rf "$BUILD_DIR" "$STAGING"
mkdir -p "$BUILD_DIR" "$STAGING"

echo "==> Configuring MLX ${VERSION} (${VARIANT})"
# Configure triggers `FetchContent_MakeAvailable` for metal_cpp / json /
# fmt, which CMake implements via a recursive `cmake --build` of a tiny
# sub-project per dep. Those sub-builds inherit `CMAKE_BUILD_PARALLEL_LEVEL`
# and race on FetchContent's download → extract → rename → stamp-touch
# pipeline when run in parallel — observed as `getcwd: cannot access
# parent directories` followed by `cd: <dir>/_deps: No such file or
# directory` (FetchContent renames out from under a sub-shell that the
# parallel make spawned). Pin the env to 1 only for this invocation;
# the main MLX build below still runs at NCPU jobs.
CMAKE_BUILD_PARALLEL_LEVEL=1 cmake \
  -S "$MLX_SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$STAGING" \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  "-DMLX_METAL_JIT=${METAL_JIT}"

NCPU="$(sysctl -n hw.ncpu)"

echo "==> Building with ${NCPU} jobs"
cmake --build "$BUILD_DIR" --parallel "$NCPU"

echo "==> Installing into ${STAGING}"
cmake --install "$BUILD_DIR"

# MLX's Metal device loader looks for mlx.metallib colocated with the
# binary. cmake --install places it under lib/ (via the install rules
# in mlx/backend/metal/CMakeLists.txt); sanity-check both artefacts
# before we rename staging into place.
for f in "lib/libmlx.a" "lib/mlx.metallib"; do
  if [[ ! -f "${STAGING}/${f}" ]]; then
    echo "error: expected ${STAGING}/${f} after cmake --install (build is incomplete)" >&2
    exit 1
  fi
done

# Atomic publish: rename staging to prefix. If a racing build already
# populated PREFIX, `mv` refuses to overwrite the non-empty target and
# exits non-zero — defer to the winner, our trap cleans STAGING up.
echo "==> Publishing to ${PREFIX}"
if ! mv "$STAGING" "$PREFIX"; then
  if [[ -f "${PREFIX}/lib/libmlx.a" && -f "${PREFIX}/lib/mlx.metallib" ]]; then
    echo "==> ${PREFIX} already populated by a concurrent build; keeping it"
  else
    echo "error: failed to publish staging to ${PREFIX}" >&2
    exit 1
  fi
fi

echo "==> Done: ${PREFIX}"
