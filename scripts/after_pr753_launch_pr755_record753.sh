#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs"

exec env \
  WAIT_LOG="${WAIT_LOG:-$LOG_DIR/h200_upstream_pr753_proxy600_timed_nocompile_seed1337.txt}" \
  WAIT_PATTERN="${WAIT_PATTERN:-^elapsed=}" \
  TARGET_LABEL="${TARGET_LABEL:-pr755_record753}" \
  TARGET_SCRIPT="${TARGET_SCRIPT:-$ROOT_DIR/scripts/icrn_h200_pr755_record753_ladder.sh}" \
  TARGET_ENV_ASSIGNMENTS="${TARGET_ENV_ASSIGNMENTS:-MAX_SHARDS=${MAX_SHARDS:-1} TIMED_MODE=${TIMED_MODE:-1} SEED=${SEED:-42} SKIP_COMPLETED=${SKIP_COMPLETED:-0}}" \
  bash "$ROOT_DIR/scripts/after_log_launch_script.sh"
