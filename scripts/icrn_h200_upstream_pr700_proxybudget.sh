#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/h200_proxy_budget.sh"

SEED="${SEED:-1337}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-$H200_PROXY_TRAIN_LIMIT_SECONDS}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"
COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
RUN_ID="${RUN_ID:-h200_upstream_pr700_proxybudget_nocompile_seed${SEED}}"

h200_proxy_budget_note
h200_proxy_guard_train_launch \
  "${ITERATIONS:-$H100_PROXY_REFERENCE_STEPS}" \
  "$MAX_WALLCLOCK_SECONDS" \
  "${ALLOW_OUT_OF_BUDGET_DEV_RUN:-0}"

echo "Launching PR700 H200 dev proxy with MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}" >&2

exec env \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  COMPILE_ENABLED="$COMPILE_ENABLED" \
  bash "$ROOT_DIR/scripts/icrn_h200_upstream_pr700_proxy.sh"
