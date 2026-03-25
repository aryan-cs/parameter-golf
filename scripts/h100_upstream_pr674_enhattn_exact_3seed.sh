#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 2045 1337 42; do
  run_suffix=""
  if [[ "${TIMED_MODE:-0}" == "1" ]]; then
    run_suffix="_timed"
  fi
  if [[ "${COMPILE_ENABLED:-1}" == "0" ]]; then
    run_suffix="${run_suffix}_nocompile"
  fi
  echo "=== seed $seed ==="
  SEED="$seed" RUN_ID="${RUN_ID:-h100_upstream_pr674_enhattn${run_suffix}_seed${seed}}" bash "$ROOT_DIR/scripts/h100_upstream_pr674_enhattn_exact.sh"
done
