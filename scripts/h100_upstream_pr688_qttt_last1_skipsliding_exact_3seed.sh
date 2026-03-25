#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-2045 2046 2047})

for seed in "${SEEDS[@]}"; do
  echo "=== SEED ${seed} ==="
  TIMED_MODE="${TIMED_MODE:-1}" \
  COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
  SEED="${seed}" \
  RUN_ID="${RUN_ID_PREFIX:-h100_upstream_pr688_timed_nocompile_qttt_last1_skipsliding}_seed${seed}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr688_qttt_last1_skipsliding_exact.sh"
done
