#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv-mac"
PYTHON_VERSION="${MAC_PROXY_PYTHON_VERSION:-3.12}"

uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
uv pip install --python "$VENV_PATH/bin/python" \
  mlx \
  numpy \
  sentencepiece \
  huggingface-hub \
  datasets \
  tqdm \
  tiktoken

echo "mac proxy env ready: $VENV_PATH"
echo "python: $("$VENV_PATH/bin/python" --version)"
