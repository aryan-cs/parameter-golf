#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-1337 42 2025})

for seed in "${SEEDS[@]}"; do
  run_suffix=""
  if [[ "${TIMED_MODE:-0}" == "1" ]]; then
    run_suffix="_timed"
  fi
  SEED="$seed" RUN_ID="${RUN_ID:-h100_upstream_pr676${run_suffix}_seed${seed}}" bash "$ROOT_DIR/scripts/h100_upstream_pr676_exact.sh"
done
