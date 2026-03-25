#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-13 1111 1337})

for seed in "${SEEDS[@]}"; do
  SEED="$seed" RUN_ID="h100_upstream_pr684_seed${seed}" bash "$ROOT_DIR/scripts/h100_upstream_pr684_exact.sh"
done
