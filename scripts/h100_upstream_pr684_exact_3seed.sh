#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=(${SEEDS:-13 1111 1337})

for seed in "${SEEDS[@]}"; do
  run_suffix=""
  if [[ "${TIMED_MODE:-0}" == "1" ]]; then
    run_suffix="_timed"
  fi
  if [[ "${USE_COMPILE:-1}" == "0" ]]; then
    run_suffix="${run_suffix}_nocompile"
  fi
  SEED="$seed" RUN_ID="${RUN_ID:-h100_upstream_pr684${run_suffix}_seed${seed}}" bash "$ROOT_DIR/scripts/h100_upstream_pr684_exact.sh"
done
