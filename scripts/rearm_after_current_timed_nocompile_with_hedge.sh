#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
SEED="${SEED:-1337}"
UPSTREAM_WAIT_PATTERN="${UPSTREAM_WAIT_PATTERN:-^final_int6_(roundtrip|sliding_window)_exact val_loss:|^elapsed=}"
NGRAM_WAIT_PATTERN="${NGRAM_WAIT_PATTERN:-^final_ngram_eval_exact val_loss:|^elapsed=}"

cd "$ROOT_DIR"
pkill -f '/scripts/after_log_launch_script.sh' || true

CURRENT_LOG="$LOG_DIR/h200_upstream_pr753_proxy600_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR758_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr758_proxy600_timed_nocompile_seed${SEED}.txt"
PR755_RECORD753_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_pr755_record753_smoke_smoke1_seed42.txt"
UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_LOG="$LOG_DIR/h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_seed${SEED}.txt"
UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_ETA05_LOG="$LOG_DIR/h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_eta05_seed${SEED}.txt"
UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_ETA20_LOG="$LOG_DIR/h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_eta20_seed${SEED}.txt"

WAIT_LOG="$CURRENT_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr758_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr758_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR758_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr758_proxy600_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR758_TIMED_NOCOMPILE_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="pr755_record753_smoke" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_pr755_record753_ladder.sh" \
NEXT_LOG_PATH="$PR755_RECORD753_SMOKE_LOG" \
NEXT_TARGET_ENV_ASSIGNMENTS="MAX_SHARDS=1 TIMED_MODE=1 SEED=42 SKIP_COMPLETED=1" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_pr753_launch_upstream_pr758_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$PR755_RECORD753_SMOKE_LOG" \
WAIT_PATTERN="$NGRAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr688_timed_nocompile_skipsliding_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr688_skipsliding_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_LOG" \
TARGET_RUN_ID="h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr688_timed_nocompile_skipsliding_eta05_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr688_skipsliding_eta05_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_ETA05_LOG" \
NEXT_TARGET_RUN_ID="h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_eta05_seed${SEED}" \
NEXT_TARGET_SEED="$SEED" \
NEXT_TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0 MIXER_ETA=0.05" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_pr755_record753_smoke_launch_upstream_pr688_skipsliding.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_ETA05_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr688_timed_nocompile_skipsliding_eta20_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr688_skipsliding_eta20_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR688_TIMED_NOCOMPILE_SKIPSLIDING_ETA20_LOG" \
TARGET_RUN_ID="h200_upstream_pr688_proxy600_timed_nocompile_skipsliding_eta20_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0 MIXER_ETA=0.20" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr688_skipsliding_eta05_launch_upstream_pr688_skipsliding_eta20.log 2>&1 < /dev/null &
