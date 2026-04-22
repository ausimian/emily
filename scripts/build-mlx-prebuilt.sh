#!/usr/bin/env bash
# Build an MLX prebuilt tarball (libmlx.a + mlx.metallib + headers) suitable
# for upload as a GitHub release asset on ausimian/emily.
#
# Usage:
#   build-mlx-prebuilt.sh <mlx_src_dir> <version> <jit 0|1>
#
# Produces in the current directory:
#   mlx-<version>-macos-<arch>-<variant>.tar.gz
#   mlx-<version>-macos-<arch>-<variant>.tar.gz.sha256
# where <arch> is arm64 on Apple Silicon and <variant> is jit|aot.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <mlx_src_dir> <version> <jit 0|1>" >&2
  exit 2
fi

MLX_SRC_DIR="$1"
VERSION="$2"
JIT="$3"

case "$JIT" in
  0) VARIANT="aot"; METAL_JIT="OFF" ;;
  1) VARIANT="jit"; METAL_JIT="ON"  ;;
  *) echo "error: jit must be 0 or 1 (got: $JIT)" >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: prebuilts are macOS-only (uname -s = $(uname -s))" >&2
  exit 2
fi

case "$(uname -m)" in
  arm64)  ARCH="arm64"  ;;
  x86_64) ARCH="x86_64" ;;
  *) echo "error: unsupported arch $(uname -m)" >&2; exit 2 ;;
esac

if [[ ! -d "$MLX_SRC_DIR" ]]; then
  echo "error: MLX source directory not found: $MLX_SRC_DIR" >&2
  exit 2
fi

# Resolve Metal toolchain. CommandLineTools alone can't run `xcrun -sdk
# macosx metal`; if the default developer dir lacks it, fall back to
# Xcode.app. This mirrors the developer_dir_env/0 logic that used to
# live in mix.exs.
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

NAME="mlx-${VERSION}-macos-${ARCH}-${VARIANT}"
WORK_DIR="$(pwd)"
BUILD_DIR="${WORK_DIR}/.mlx-prebuilt-build/${NAME}"
STAGE_DIR="${WORK_DIR}/.mlx-prebuilt-stage/${NAME}"
TARBALL="${WORK_DIR}/${NAME}.tar.gz"

rm -rf "$BUILD_DIR" "$STAGE_DIR"
mkdir -p "$BUILD_DIR" "$STAGE_DIR"

echo "==> Configuring MLX ${VERSION} (${VARIANT})"
cmake \
  -S "$MLX_SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$STAGE_DIR" \
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

echo "==> Installing into ${STAGE_DIR}"
cmake --install "$BUILD_DIR"

# MLX's Metal device loader looks for mlx.metallib colocated with the
# binary. cmake --install places it under lib/ (via the install rules
# in mlx/backend/metal/CMakeLists.txt); sanity-check that the shipped
# tarball actually contains both artefacts.
for f in "lib/libmlx.a" "lib/mlx.metallib"; do
  if [[ ! -f "${STAGE_DIR}/${f}" ]]; then
    echo "error: expected ${STAGE_DIR}/${f} after cmake --install (build is incomplete)" >&2
    exit 1
  fi
done

echo "==> Packaging ${TARBALL}"
rm -f "$TARBALL" "${TARBALL}.sha256"
tar czf "$TARBALL" -C "${STAGE_DIR}/.." "$NAME"

# Side-car sha256 matches the `shasum -a 256 <file>` output format used
# throughout the macOS toolchain; mix.exs only reads the first 64 hex
# chars so either format is fine.
shasum -a 256 "$TARBALL" > "${TARBALL}.sha256"

echo "==> Done"
ls -lh "$TARBALL" "${TARBALL}.sha256"
cat "${TARBALL}.sha256"
