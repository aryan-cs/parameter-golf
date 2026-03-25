#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

PROXY_TRAIN_LOG="${PROXY_TRAIN_LOG:-$LOG_DIR/h200_ttt_h100proxy7185_seed1337.txt}"
PROXY_ARTIFACT_PT="${PROXY_ARTIFACT_PT:-$RUN_DIR/final_model_h100proxy7185_seed1337.pt}"
PROXY_ARTIFACT_INT6="${PROXY_ARTIFACT_INT6:-$RUN_DIR/final_model_h100proxy7185_seed1337.int6.ptz}"
PROXY_RECORD674_LOG="${PROXY_RECORD674_LOG:-$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_seed1337.txt}"
PROXY_RECORD674_LAM18_SMOKE_LOG="${PROXY_RECORD674_LAM18_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185_seed1337_smoke.txt}"
PROXY_RECORD674_LAM22_SMOKE_LOG="${PROXY_RECORD674_LAM22_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185_seed1337_smoke.txt}"
PROXY_RECORD674_MIN3_SMOKE_LOG="${PROXY_RECORD674_MIN3_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185_seed1337_smoke.txt}"
PROXY_CONF07_LOG="${PROXY_CONF07_LOG:-$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_seed1337.txt}"

cd "$ROOT_DIR"
source .venv/bin/activate

while true; do
  if [[ -f "$PROXY_TRAIN_LOG" ]] && rg -q '^(Serialized model int6\+lzma:|Total submission size int6\+lzma:|final_int6_roundtrip val_loss:)' "$PROXY_TRAIN_LOG"; then
    break
  fi
  sleep 60
done

for artifact_path in "$RUN_DIR/final_model.pt" "$RUN_DIR/final_model.int6.ptz"; do
  while [[ ! -f "$artifact_path" ]]; do
    sleep 5
  done
done

cp "$RUN_DIR/final_model.pt" "$PROXY_ARTIFACT_PT"
cp "$RUN_DIR/final_model.int6.ptz" "$PROXY_ARTIFACT_INT6"

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
CANDIDATE="record674_proxy7185" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
LOG_PATH="$PROXY_RECORD674_LAM18_SMOKE_LOG" \
NGRAM_LAMBDA="0.18" \
MAX_WINDOWS="128" \
CANDIDATE="record674_proxy7185" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
LOG_PATH="$PROXY_RECORD674_LAM22_SMOKE_LOG" \
NGRAM_LAMBDA="0.22" \
MAX_WINDOWS="128" \
CANDIDATE="record674_proxy7185" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
LOG_PATH="$PROXY_RECORD674_MIN3_SMOKE_LOG" \
MIN_COUNT="3" \
MAX_WINDOWS="128" \
CANDIDATE="record674_proxy7185" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
LOG_PATH="$PROXY_CONF07_LOG" \
CANDIDATE="record659_conf07" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

python "$ROOT_DIR/scripts/record_push_status.py" --seed 1337
