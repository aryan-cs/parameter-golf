#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
SEED="${SEED:-1337}"
RUN_NEARBY_SMOKES="${RUN_NEARBY_SMOKES:-0}"
RUN_CONF07_TAIL="${RUN_CONF07_TAIL:-0}"
UPSTREAM_WAIT_PATTERN="${UPSTREAM_WAIT_PATTERN:-^final_int6_sliding_window_ngram5_exact val_loss:}"
UPSTREAM_PR676_WAIT_PATTERN="${UPSTREAM_PR676_WAIT_PATTERN:-^legal_ttt_exact val_loss:}"
UPSTREAM_PR685_MEANPROB_WAIT_PATTERN="${UPSTREAM_PR685_MEANPROB_WAIT_PATTERN:-^final_int8_zlib_roundtrip_exact val_loss:}"
UPSTREAM_PR685_PHASE1_WAIT_PATTERN="${UPSTREAM_PR685_PHASE1_WAIT_PATTERN:-^final_int8_zlib_roundtrip_exact val_loss:}"
UPSTREAM_PR684_WAIT_PATTERN="${UPSTREAM_PR684_WAIT_PATTERN:-^final_int6_sliding_window_exact val_loss:}"

launch_after_record674_arch() {
  local wait_log="$1"
  local arch="$2"
  local wait_pattern="${3:-^final_(ngram_eval_exact|int6_sliding_window_ngram[0-9]+_exact) val_loss:}"
  local suffix=""
  if [[ "$arch" != "baseline" ]]; then
    suffix="_${arch}"
  fi
  local proxy_train_log="$LOG_DIR/h200_ttt_h100proxy7185${suffix}_seed${SEED}.txt"
  local proxy_artifact_pt="$RUN_DIR/final_model_h100proxy7185${suffix}_seed${SEED}.pt"
  local proxy_artifact_int6="$RUN_DIR/final_model_h100proxy7185${suffix}_seed${SEED}.int6.ptz"
  local record674_log="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185${suffix}_seed${SEED}.txt"
  local lam18_smoke_log="$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185${suffix}_seed${SEED}_smoke.txt"
  local lam22_smoke_log="$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185${suffix}_seed${SEED}_smoke.txt"
  local min3_smoke_log="$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185${suffix}_seed${SEED}_smoke.txt"
  local conf07_log="$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185${suffix}_seed${SEED}.txt"
  setsid env \
    WAIT_RECORD674_LOG="$wait_log" \
    WAIT_PATTERN="$wait_pattern" \
    TARGET_ARCH_CANDIDATE="$arch" \
    TARGET_PROXY_TRAIN_LOG="$proxy_train_log" \
    TARGET_PROXY_ARTIFACT_PT="$proxy_artifact_pt" \
    TARGET_PROXY_ARTIFACT_INT6="$proxy_artifact_int6" \
    TARGET_RECORD674_LOG="$record674_log" \
    TARGET_RECORD674_LAM18_SMOKE_LOG="$lam18_smoke_log" \
    TARGET_RECORD674_LAM22_SMOKE_LOG="$lam22_smoke_log" \
    TARGET_RECORD674_MIN3_SMOKE_LOG="$min3_smoke_log" \
    TARGET_CONF07_LOG="$conf07_log" \
    RUN_NEARBY_SMOKES="$RUN_NEARBY_SMOKES" \
    RUN_CONF07_TAIL="$RUN_CONF07_TAIL" \
    bash "$ROOT_DIR/scripts/after_record674_launch_arch.sh" >/tmp/h200_after_${arch}_record674_queue.log 2>&1 < /dev/null &
}

cd "$ROOT_DIR"
pkill -f 'after_proxy_train_run_record674_then_conf07|after_record674_launch_arch|after_xsa11_proxy_queue_launch_podracing674_xsa11|after_log_launch_script' || true

BASE_PROXY_TRAIN_LOG="$LOG_DIR/h200_ttt_h100proxy7185_seed${SEED}.txt"
BASE_PROXY_ARTIFACT_PT="$RUN_DIR/final_model_h100proxy7185_seed${SEED}.pt"
BASE_PROXY_ARTIFACT_INT6="$RUN_DIR/final_model_h100proxy7185_seed${SEED}.int6.ptz"
BASE_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_seed${SEED}.txt"
BASE_RECORD674_LAM18_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185_seed${SEED}_smoke.txt"
BASE_RECORD674_LAM22_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185_seed${SEED}_smoke.txt"
BASE_RECORD674_MIN3_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185_seed${SEED}_smoke.txt"
BASE_CONF07_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_seed${SEED}.txt"
UPSTREAM_PR674_LOG="$LOG_DIR/h200_upstream_pr674_proxy7185_seed${SEED}.txt"
UPSTREAM_PR676_LOG="$LOG_DIR/h200_upstream_pr676_proxy7185_seed${SEED}.txt"
UPSTREAM_PR685_MEANPROB_LOG="$LOG_DIR/h200_upstream_pr685_meanprob_proxy7185_seed${SEED}.txt"
UPSTREAM_PR685_PHASE1_LOG="$LOG_DIR/h200_upstream_pr685_phase1_proxy7185_seed${SEED}.txt"
UPSTREAM_PR684_LOG="$LOG_DIR/h200_upstream_pr684_proxy6555_seed${SEED}.txt"

PROXY_TRAIN_LOG="$BASE_PROXY_TRAIN_LOG" \
PROXY_ARTIFACT_PT="$BASE_PROXY_ARTIFACT_PT" \
PROXY_ARTIFACT_INT6="$BASE_PROXY_ARTIFACT_INT6" \
PROXY_RECORD674_LOG="$BASE_RECORD674_LOG" \
PROXY_RECORD674_LAM18_SMOKE_LOG="$BASE_RECORD674_LAM18_SMOKE_LOG" \
PROXY_RECORD674_LAM22_SMOKE_LOG="$BASE_RECORD674_LAM22_SMOKE_LOG" \
PROXY_RECORD674_MIN3_SMOKE_LOG="$BASE_RECORD674_MIN3_SMOKE_LOG" \
PROXY_CONF07_LOG="$BASE_CONF07_LOG" \
RUN_NEARBY_SMOKES="$RUN_NEARBY_SMOKES" \
RUN_CONF07_TAIL="$RUN_CONF07_TAIL" \
setsid bash "$ROOT_DIR/scripts/after_proxy_train_run_record674_then_conf07.sh" >/tmp/h200_after_baseline_proxy_train_record674.log 2>&1 < /dev/null &

WAIT_LOG="$BASE_RECORD674_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
NEXT_WAIT_LOG="$UPSTREAM_PR674_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr676_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr676_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR676_LOG" \
NEXT_RUN_ID="h200_upstream_pr676_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_baseline_record674_launch_upstream_pr674.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR676_LOG" \
WAIT_PATTERN="$UPSTREAM_PR676_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr685_meanprob_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_meanprob_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR685_MEANPROB_LOG" \
TARGET_RUN_ID="h200_upstream_pr685_meanprob_proxy7185_seed${SEED}" \
TARGET_SEED="$SEED" \
NEXT_WAIT_LOG="$UPSTREAM_PR685_MEANPROB_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_PR685_MEANPROB_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr685_phase1_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr685_phase1_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR685_PHASE1_LOG" \
NEXT_RUN_ID="h200_upstream_pr685_phase1_proxy7185_seed${SEED}" \
NEXT_SEED="$SEED" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr676_launch_upstream_pr685_meanprob.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR685_PHASE1_LOG" \
WAIT_PATTERN="$UPSTREAM_PR685_PHASE1_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr684_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr684_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR684_LOG" \
TARGET_RUN_ID="h200_upstream_pr684_proxy6555_seed${SEED}" \
TARGET_SEED="$SEED" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr685_phase1_launch_upstream_pr684.log 2>&1 < /dev/null &

POD_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_podracing674_seed${SEED}.txt"
POD_SWIGLU_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_podracing674_swiglu_seed${SEED}.txt"
SWIGLU676_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_swiglu676_seed${SEED}.txt"
XSA11_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_xsa11_seed${SEED}.txt"

launch_after_record674_arch "$UPSTREAM_PR684_LOG" "podracing674" "$UPSTREAM_PR684_WAIT_PATTERN"
launch_after_record674_arch "$POD_RECORD674_LOG" "podracing674_swiglu"
launch_after_record674_arch "$POD_SWIGLU_RECORD674_LOG" "swiglu676"
launch_after_record674_arch "$SWIGLU676_RECORD674_LOG" "xsa11"
launch_after_record674_arch "$XSA11_RECORD674_LOG" "podracing674_xsa11"
