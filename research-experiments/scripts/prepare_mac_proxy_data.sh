#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv-mac/bin/python"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
source "$SCRIPT_DIR/mac_proxy_uv_env.sh"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "missing mac proxy environment at $VENV_PYTHON"
  echo "run: bash research-experiments/scripts/setup_mac_proxy_env.sh"
  exit 1
fi

cd "$REPO_ROOT/research-experiments"
uv run --python "$VENV_PYTHON" python scripts/prepare_challenge_data.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
