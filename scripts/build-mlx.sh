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

# Keep the scratch build dir alongside the install prefix so `--force`
# from mix.exs (which rm -rf's the install dir) doesn't orphan it.
BUILD_DIR="${PREFIX}.build"

rm -rf "$BUILD_DIR" "$PREFIX"
mkdir -p "$BUILD_DIR" "$PREFIX"

echo "==> Configuring MLX ${VERSION} (${VARIANT})"
cmake \
  -S "$MLX_SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
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

echo "==> Installing into ${PREFIX}"
cmake --install "$BUILD_DIR"

# MLX's Metal device loader looks for mlx.metallib colocated with the
# binary. cmake --install places it under lib/ (via the install rules
# in mlx/backend/metal/CMakeLists.txt); sanity-check both artefacts.
for f in "lib/libmlx.a" "lib/mlx.metallib"; do
  if [[ ! -f "${PREFIX}/${f}" ]]; then
    echo "error: expected ${PREFIX}/${f} after cmake --install (build is incomplete)" >&2
    exit 1
  fi
done

# Scratch build is large (~1.5 GB on the aot lane). mix.exs doesn't
# need it once the install dir is populated.
rm -rf "$BUILD_DIR"

echo "==> Done: ${PREFIX}"
