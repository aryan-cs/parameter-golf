#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-42 137 3})

for seed in "${SEEDS[@]}"; do
  echo "=== PR755 exact seed ${seed} ==="
  SEED="$seed" RUN_ID="h100_upstream_pr755_exact_seed${seed}" \
    bash "$ROOT_DIR/scripts/h100_upstream_pr755_exact.sh"
done
