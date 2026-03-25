#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

PROXY_CONF07_LOG="${PROXY_CONF07_LOG:-$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_seed1337.txt}"
PROXY_ARTIFACT_PT="${PROXY_ARTIFACT_PT:-$RUN_DIR/final_model_h100proxy7185_seed1337.pt}"
PROXY_ARTIFACT_INT6="${PROXY_ARTIFACT_INT6:-$RUN_DIR/final_model_h100proxy7185_seed1337.int6.ptz}"

cd "$ROOT_DIR"
source .venv/bin/activate

while ! rg -q "final_ngram_eval_exact" "$PROXY_CONF07_LOG" 2>/dev/null; do
  sleep 60
done

ARTIFACT_PATH="$PROXY_ARTIFACT_INT6" \
TEMPLATE_PATH="$PROXY_ARTIFACT_PT" \
TRAIN_GPT_PATH="$RUN_DIR/train_gpt.py" \
CANDIDATE="record674_proxy7185" \
bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"

python "$ROOT_DIR/scripts/record_push_status.py" --seed 1337
