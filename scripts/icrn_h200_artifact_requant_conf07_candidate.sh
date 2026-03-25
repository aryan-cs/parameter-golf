#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

REQUANT_SLUG="${REQUANT_SLUG:-plus}"
INNER_CANDIDATE="${INNER_CANDIDATE:-record659_conf07_smoke}"
CLIP_PCTS="${CLIP_PCTS:-0.998,0.9990,0.99925,0.9995,0.99975,0.9999,0.99999,1.0}"
TEMPLATE_PATH="${TEMPLATE_PATH:-$RUN_DIR/final_model.pt}"
REFERENCE_ARTIFACT_PATH="${REFERENCE_ARTIFACT_PATH:-$RUN_DIR/final_model.int6.ptz}"
TRAIN_GPT_PATH="${TRAIN_GPT_PATH:-$RUN_DIR/train_gpt.py}"
ARTIFACT_OUT="${ARTIFACT_OUT:-$RUN_DIR/final_model_requant_${REQUANT_SLUG}.int6.ptz}"
QUANT_LOG="${QUANT_LOG:-$LOG_DIR/h200_artifact_requant_${REQUANT_SLUG}.txt}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_requant_${REQUANT_SLUG}.txt}"

cd "$ROOT_DIR"
source .venv/bin/activate

python "$ROOT_DIR/scripts/requantize_artifact_gptq.py" \
  --train-gpt-path "$TRAIN_GPT_PATH" \
  --template-path "$TEMPLATE_PATH" \
  --output-artifact "$ARTIFACT_OUT" \
  --reference-artifact "$REFERENCE_ARTIFACT_PATH" \
  --int6-clip-pcts "$CLIP_PCTS" \
  --verify-roundtrip | tee "$QUANT_LOG"

env \
  ARTIFACT_PATH="$ARTIFACT_OUT" \
  TEMPLATE_PATH="$TEMPLATE_PATH" \
  TRAIN_GPT_PATH="$TRAIN_GPT_PATH" \
  CANDIDATE="$INNER_CANDIDATE" \
  LOG_PATH="$LOG_PATH" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"
