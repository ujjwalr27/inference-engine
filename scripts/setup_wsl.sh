#!/usr/bin/env bash
# One-time WSL2 Ubuntu setup for gpt2-engine. Safe to re-run.
# Usage (inside WSL): bash scripts/setup_wsl.sh
set -euo pipefail

LIBTORCH_VERSION="2.10.0"
LIBTORCH_DIR="$HOME/libtorch-${LIBTORCH_VERSION}-cpu"
VENV_DIR="$HOME/venvs/gpt2-engine"
PYTHON_VERSION="3.12"

echo "==> apt packages (asks for your sudo password)"
sudo apt-get update
sudo apt-get install -y build-essential clang cmake ninja-build git unzip curl pkg-config

echo "==> LibTorch ${LIBTORCH_VERSION} CPU -> ${LIBTORCH_DIR}"
if [[ ! -f "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "${tmp}/libtorch.zip" \
    "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
  unzip -q "${tmp}/libtorch.zip" -d "${tmp}"
  rm -rf "${LIBTORCH_DIR}"
  mv "${tmp}/libtorch" "${LIBTORCH_DIR}"
  rm -rf "${tmp}"
else
  echo "    already present"
fi

echo "==> Rust (for tokenizers-cpp, Phase 2)"
if ! command -v cargo >/dev/null 2>&1 && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi

echo "==> uv + Python ${PYTHON_VERSION} venv -> ${VENV_DIR}"
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
[[ -d "${VENV_DIR}" ]] || uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
VIRTUAL_ENV="${VENV_DIR}" uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==${LIBTORCH_VERSION}"
VIRTUAL_ENV="${VENV_DIR}" uv pip install transformers safetensors pandas matplotlib

echo "==> shell profile"
PROFILE_LINE="export LIBTORCH_DIR=\"${LIBTORCH_DIR}\""
grep -qxF "${PROFILE_LINE}" "$HOME/.bashrc" || echo "${PROFILE_LINE}" >> "$HOME/.bashrc"

echo
echo "Done. Open a new shell (or 'source ~/.bashrc'), then run: bash scripts/build.sh"
