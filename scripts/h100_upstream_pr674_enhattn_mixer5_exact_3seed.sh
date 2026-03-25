#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMED_MODE="${TIMED_MODE:-0}"
COMPILE_ENABLED="${COMPILE_ENABLED:-1}"

for seed in 1337 42 2025; do
  run_suffix=""
  if [[ "$TIMED_MODE" == "1" ]]; then
    run_suffix="_timed"
  fi
  if [[ "$COMPILE_ENABLED" == "0" ]]; then
    run_suffix="${run_suffix}_nocompile"
  fi
  SEED="$seed" \
  TIMED_MODE="$TIMED_MODE" \
  COMPILE_ENABLED="$COMPILE_ENABLED" \
  RUN_ID="${RUN_ID:-h100_upstream_pr674_enhattn_mixer5${run_suffix}_seed${seed}}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr674_enhattn_mixer5_exact.sh"
done
