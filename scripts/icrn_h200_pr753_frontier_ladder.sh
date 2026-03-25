#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"

ARTIFACT_PATH="${ARTIFACT_PATH:-$RECORD_DIR/final_model_h100proxy7185_seed1337.int6.ptz}"
TEMPLATE_PATH="${TEMPLATE_PATH:-$RECORD_DIR/final_model_h100proxy7185_seed1337.pt}"
TRAIN_GPT_PATH="${TRAIN_GPT_PATH:-$RECORD_DIR/train_gpt.py}"
SEED="${SEED:-1337}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
RUN_LOCAL_ABLATIONS="${RUN_LOCAL_ABLATIONS:-0}"

cd "$ROOT_DIR"

run_artifact_candidate() {
  local candidate="$1"
  shift
  env \
    ARTIFACT_PATH="$ARTIFACT_PATH" \
    TEMPLATE_PATH="$TEMPLATE_PATH" \
    TRAIN_GPT_PATH="$TRAIN_GPT_PATH" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    CANDIDATE="$candidate" \
    "$@" \
    bash scripts/icrn_h200_artifact_ngram_candidate.sh
}

env \
  COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
  TIMED_MODE="${TIMED_MODE:-1}" \
  SEED="$SEED" \
  bash scripts/icrn_h200_upstream_pr753_proxy.sh

if [[ "$RUN_LOCAL_ABLATIONS" != "1" ]]; then
  exit 0
fi

run_artifact_candidate "record753_proxy7185"
run_artifact_candidate \
  "record753_proxy7185" \
  NGRAM_LAMBDA=0.40 \
  NGRAM_ADAPTIVE_ALPHA=0 \
  LOG_PATH="$RECORD_DIR/logs/h200_artifact_ngram_record753_fixed40.txt"
run_artifact_candidate \
  "record753_proxy7185" \
  NGRAM_LAMBDA=0.40 \
  NGRAM_ADAPTIVE_ALPHA=0 \
  NGRAM_MIN_ORDER=7 \
  LOG_PATH="$RECORD_DIR/logs/h200_artifact_ngram_record753_nobackoff_fixed40.txt"
