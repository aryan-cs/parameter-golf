#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
SEED="${SEED:-1337}"
UPSTREAM_WAIT_PATTERN="${UPSTREAM_WAIT_PATTERN:-^final_int6_sliding_window_ngram5_exact val_loss:}"
UPSTREAM_PR674_ENHATTN_WAIT_PATTERN="${UPSTREAM_PR674_ENHATTN_WAIT_PATTERN:-^final_int6_sliding_window_ngram5_exact val_loss:}"
UPSTREAM_PR676_WAIT_PATTERN="${UPSTREAM_PR676_WAIT_PATTERN:-^legal_ttt_exact val_loss:}"
UPSTREAM_PR685_POWMEAN4_WAIT_PATTERN="${UPSTREAM_PR685_POWMEAN4_WAIT_PATTERN:-^final_int8_zlib_roundtrip_exact val_loss:}"
UPSTREAM_PR685_MEANPROB_WAIT_PATTERN="${UPSTREAM_PR685_MEANPROB_WAIT_PATTERN:-^final_int8_zlib_roundtrip_exact val_loss:}"
UPSTREAM_PR685_PHASE1_WAIT_PATTERN="${UPSTREAM_PR685_PHASE1_WAIT_PATTERN:-^final_int8_zlib_roundtrip_exact val_loss:}"
UPSTREAM_PR684_WAIT_PATTERN="${UPSTREAM_PR684_WAIT_PATTERN:-^final_int6_sliding_window_exact val_loss:}"

cd "$ROOT_DIR"
pkill -f 'after_proxy_train_run_record674_then_conf07|after_record674_launch_arch|after_xsa11_proxy_queue_launch_podracing674_xsa11|after_log_launch_script' || true

UPSTREAM_PR674_TIMED_LOG="$LOG_DIR/h200_upstream_pr674_proxy7185_timed_seed${SEED}.txt"
UPSTREAM_PR674_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_HEDGEMIX_LOG="$LOG_DIR/h200_upstream_pr674_hedgemix_proxy7185_seed${SEED}.txt"
UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_hedgemix_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_ENHATTN_TIMED_LOG="$LOG_DIR/h200_upstream_pr674_enhattn_proxy7185_timed_seed${SEED}.txt"
UPSTREAM_PR674_ENHATTN_LOG="$LOG_DIR/h200_upstream_pr674_enhattn_proxy7185_seed${SEED}.txt"
UPSTREAM_PR676_TIMED_LOG="$LOG_DIR/h200_upstream_pr676_proxy7185_timed_seed${SEED}.txt"
UPSTREAM_PR676_LOG="$LOG_DIR/h200_upstream_pr676_proxy7185_seed${SEED}.txt"
UPSTREAM_PR685_POWMEAN4_LOG="$LOG_DIR/h200_upstream_pr685_powmean4_proxy7185_seed${SEED}.txt"
UPSTREAM_PR685_MEANPROB_LOG="$LOG_DIR/h200_upstream_pr685_meanprob_proxy7185_seed${SEED}.txt"
UPSTREAM_PR685_PHASE1_LOG="$LOG_DIR/h200_upstream_pr685_phase1_proxy7185_seed${SEED}.txt"
UPSTREAM_PR684_TIMED_LOG="$LOG_DIR/h200_upstream_pr684_proxy6555_timed_seed${SEED}.txt"
UPSTREAM_PR684_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr684_proxy6555_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR684_LOG="$LOG_DIR/h200_upstream_pr684_proxy6555_seed${SEED}.txt"

WAIT_LOG="$UPSTREAM_PR674_TIMED_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_proxy7185_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR674_TIMED_NOCOMPILE_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr674_hedgemix_timed_nocompile_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_hedgemix_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG" \
NEXT_RUN_ID="h200_upstream_pr674_hedgemix_proxy7185_timed_nocompile_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_timed_launch_upstream_pr674_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr676_timed_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr676_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR676_TIMED_LOG" \
TARGET_RUN_ID="h200_upstream_pr676_proxy7185_timed_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR676_TIMED_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR676_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr674_hedgemix_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_hedgemix_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR674_HEDGEMIX_LOG" \
NEXT_RUN_ID="h200_upstream_pr674_hedgemix_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_hedgemix_timed_nocompile_launch_upstream_pr676_timed.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_HEDGEMIX_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr676_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr676_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR676_LOG" \
TARGET_RUN_ID="h200_upstream_pr676_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR676_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR676_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr674_enhattn_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_enhattn_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR674_ENHATTN_LOG" \
NEXT_RUN_ID="h200_upstream_pr674_enhattn_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_hedgemix_launch_upstream_pr676.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_ENHATTN_LOG" \
WAIT_PATTERN="$UPSTREAM_PR674_ENHATTN_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr676_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr676_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR676_LOG" \
TARGET_RUN_ID="h200_upstream_pr676_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR676_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR676_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr685_powmean4_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_powmean4_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR685_POWMEAN4_LOG" \
NEXT_RUN_ID="h200_upstream_pr685_powmean4_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_enhattn_launch_upstream_pr676.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR676_LOG" \
WAIT_PATTERN="$UPSTREAM_PR676_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr685_powmean4_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_powmean4_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR685_POWMEAN4_LOG" \
TARGET_RUN_ID="h200_upstream_pr685_powmean4_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR685_POWMEAN4_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR685_POWMEAN4_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr685_meanprob_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_meanprob_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR685_MEANPROB_LOG" \
NEXT_RUN_ID="h200_upstream_pr685_meanprob_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr676_launch_upstream_pr685_powmean4.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR685_MEANPROB_LOG" \
WAIT_PATTERN="$UPSTREAM_PR685_MEANPROB_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr685_phase1_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_phase1_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR685_PHASE1_LOG" \
TARGET_RUN_ID="h200_upstream_pr685_phase1_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR685_PHASE1_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR685_PHASE1_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr684_timed_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr684_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR684_TIMED_LOG" \
NEXT_RUN_ID="h200_upstream_pr684_proxy6555_timed_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr685_powmean4_launch_upstream_pr685_meanprob.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR684_TIMED_LOG" \
WAIT_PATTERN="$UPSTREAM_PR684_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr684_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr684_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR684_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr684_proxy6555_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 USE_COMPILE=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR684_TIMED_NOCOMPILE_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR684_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr684_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr684_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR684_LOG" \
NEXT_RUN_ID="h200_upstream_pr684_proxy6555_seed${SEED}" \
NEXT_SEED="$SEED" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr684_timed_launch_upstream_pr684_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR684_TIMED_NOCOMPILE_LOG" \
WAIT_PATTERN="$UPSTREAM_PR684_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr684_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr684_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR684_LOG" \
TARGET_RUN_ID="h200_upstream_pr684_proxy6555_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr684_timed_nocompile_launch_upstream_pr684.log 2>&1 < /dev/null &
