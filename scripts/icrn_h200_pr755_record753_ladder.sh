#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_GravityTokenizer_AblationLeverage"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"

SEED="${SEED:-42}"
MAX_SHARDS="${MAX_SHARDS:-1}"
TIMED_MODE="${TIMED_MODE:-1}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

if [[ "$MAX_SHARDS" == "0" ]]; then
  data_suffix="full"
  gravity_data_path="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0"
else
  data_suffix="smoke${MAX_SHARDS}"
  gravity_data_path="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0_smoke${MAX_SHARDS}"
fi

gravity_tokenizer_path="$ROOT_DIR/data/tokenizers/gravity_beta_1.0.model"
run_id="${RUN_ID:-h200_upstream_pr755_proxy600_${data_suffix}_seed${SEED}}"
patch_run_dir="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr755-run-${run_id}}"
patch_record_dir="$patch_run_dir/$RECORD_DIR_REL"

cd "$ROOT_DIR"

env \
  MAX_SHARDS="$MAX_SHARDS" \
  LOG_PATH="${SETUP_LOG_PATH:-$LOG_DIR/h200_upstream_pr755_setup_${data_suffix}.txt}" \
  bash scripts/icrn_h200_upstream_pr755_setup.sh

env \
  MAX_SHARDS="$MAX_SHARDS" \
  TIMED_MODE="$TIMED_MODE" \
  SEED="$SEED" \
  RUN_ID="$run_id" \
  PATCH_RUN_DIR="$patch_run_dir" \
  LOG_PATH="${TRAIN_LOG_PATH:-$LOG_DIR/${run_id}.txt}" \
  bash scripts/icrn_h200_upstream_pr755_proxy.sh

artifact_path="$patch_record_dir/final_model.int8.ptz"
template_path="$patch_record_dir/final_model.pt"
train_gpt_path="$patch_record_dir/train_gpt.py"

candidate="${EVAL_CANDIDATE:-record753_smoke}"
if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  candidate="${FULL_EVAL_CANDIDATE:-record753}"
fi

env \
  ARTIFACT_PATH="$artifact_path" \
  TEMPLATE_PATH="$template_path" \
  TRAIN_GPT_PATH="$train_gpt_path" \
  EVAL_DATA_PATH="$gravity_data_path" \
  EVAL_TOKENIZER_PATH="$gravity_tokenizer_path" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  CANDIDATE="$candidate" \
  LOG_PATH="${EVAL_LOG_PATH:-$LOG_DIR/h200_artifact_ngram_pr755_${candidate}_${data_suffix}_seed${SEED}.txt}" \
  bash scripts/icrn_h200_artifact_ngram_candidate.sh
