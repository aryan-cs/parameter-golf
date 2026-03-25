#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

CONF07_LIVE_LOG="${CONF07_LIVE_LOG:-$LOG_DIR/h200_artifact_ngram_record659_conf07.txt}"
LONG_ARTIFACT_PT="$RUN_DIR/final_model_longartifact_seed1337.pt"
LONG_ARTIFACT_INT6="$RUN_DIR/final_model_longartifact_seed1337.int6.ptz"
PROXY_ARTIFACT_PT="$RUN_DIR/final_model_h100proxy7185_seed1337.pt"
PROXY_ARTIFACT_INT6="$RUN_DIR/final_model_h100proxy7185_seed1337.int6.ptz"
LONG_CONF07_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_longartifact_seed1337.txt"
PROXY_TRAIN_LOG="$LOG_DIR/h200_ttt_h100proxy7185_seed1337.txt"
PROXY_TRAIN_LOG_BACKUP="$LOG_DIR/h200_ttt_h100proxy7185_seed1337_prev.txt"
PROXY_CONF07_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_seed1337.txt"

cd "$ROOT_DIR"
source .venv/bin/activate

while ! rg -q "final_ngram_eval_exact" "$CONF07_LIVE_LOG" 2>/dev/null; do
  sleep 60
done

mkdir -p "$LOG_DIR"

if [[ -f "$RUN_DIR/final_model.pt" && ! -f "$LONG_ARTIFACT_PT" ]]; then
  cp "$RUN_DIR/final_model.pt" "$LONG_ARTIFACT_PT"
fi
if [[ -f "$RUN_DIR/final_model.int6.ptz" && ! -f "$LONG_ARTIFACT_INT6" ]]; then
  cp "$RUN_DIR/final_model.int6.ptz" "$LONG_ARTIFACT_INT6"
fi
if [[ -f "$CONF07_LIVE_LOG" && ! -f "$LONG_CONF07_LOG" ]]; then
  cp "$CONF07_LIVE_LOG" "$LONG_CONF07_LOG"
fi
if [[ -f "$PROXY_TRAIN_LOG" && ! -f "$PROXY_TRAIN_LOG_BACKUP" ]]; then
  cp "$PROXY_TRAIN_LOG" "$PROXY_TRAIN_LOG_BACKUP"
fi

rm -f "$PROXY_TRAIN_LOG" "$PROXY_CONF07_LOG"

(
  cd "$ROOT_DIR"
  RUN_ID="h200_ttt_h100proxy7185_seed1337" \
  SEED="${SEED:-1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy.sh"
) >"$PROXY_TRAIN_LOG" 2>&1

cp "$RUN_DIR/final_model.pt" "$PROXY_ARTIFACT_PT"
cp "$RUN_DIR/final_model.int6.ptz" "$PROXY_ARTIFACT_INT6"

python "$ROOT_DIR/scripts/eval_ngram_cache_artifact.py" \
  --run-dir "$RUN_DIR" \
  --artifact-path "$PROXY_ARTIFACT_INT6" \
  --template-path "$PROXY_ARTIFACT_PT" \
  --train-gpt-path "$RUN_DIR/train_gpt.py" \
  --log-path "$PROXY_CONF07_LOG" \
  --batch-seqs 32 \
  --bigram-vocab-size 1536 \
  --value-residual 0 \
  --stride 128 \
  --ngram-lambda 0.15 \
  --ngram-max-n 5 \
  --confidence-threshold 0.7 \
  --min-count 3 \
  --packed-cache

python "$ROOT_DIR/scripts/record_push_status.py" --seed 1337
