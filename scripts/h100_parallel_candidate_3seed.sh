#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-1337 42 2025}"
CANDIDATE="${CANDIDATE:-baseline}"

for seed in $SEEDS; do
  echo "=== Starting candidate ${CANDIDATE} seed ${seed} ==="
  SEED="$seed" CANDIDATE="$CANDIDATE" \
    bash "$ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh"
done
