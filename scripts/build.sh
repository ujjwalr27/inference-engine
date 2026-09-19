#!/usr/bin/env bash
# Configure, build, and test. Build dir lives on the Linux filesystem for speed.
# Usage: bash scripts/build.sh [Release|RelWithDebInfo|Debug] [extra cmake args...]
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-RelWithDebInfo}"
shift || true
BUILD_DIR="${BUILD_DIR:-$HOME/build/gpt2-engine-${BUILD_TYPE}}"
LIBTORCH_DIR="${LIBTORCH_DIR:-$HOME/libtorch-2.10.0-cpu}"
# Compiling torch + GoogleTest headers takes ~2 GB per job, and WSL defaults to about half
# your RAM, so 2 is the safe default here. Raise it if you gave WSL more memory via .wslconfig.
JOBS="${JOBS:-2}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" \
  "$@"
cmake --build "${BUILD_DIR}" -j "${JOBS}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
"${BUILD_DIR}/hello_tensor" cpu
