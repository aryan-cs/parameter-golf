#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p cache logs runs data/datasets data/tokenizers

{
  echo "repo_root=$ROOT"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uv=$(uv --version)"
  echo "python=$(python3 --version 2>&1)"
  echo "nvidia_smi=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd ';' -)"
} | tee "logs/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).txt"

if [[ -f uv.lock ]]; then
  uv sync --extra cuda --frozen
else
  uv sync --extra cuda
fi

if [[ "${DOWNLOAD_DATASET:-1}" == "1" ]] && [[ ! -f data/tokenizers/fineweb_1024_bpe.model ]]; then
  uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS:-80}"
fi

echo "Bootstrap complete."
