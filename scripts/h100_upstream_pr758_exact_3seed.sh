#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-42 1337 2024})

for seed in "${SEEDS[@]}"; do
  echo "=== PR758 exact seed ${seed} ==="
  SEED="$seed" RUN_ID="h100_upstream_pr758_exact_seed${seed}" \
    bash "$ROOT_DIR/scripts/h100_upstream_pr758_exact.sh"
done
