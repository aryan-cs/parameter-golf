#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
CANDIDATE="${CANDIDATE:-record659}"

cd "$ROOT_DIR"
source .venv/bin/activate

STRIDE="${STRIDE:-128}"
NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
MIN_COUNT="${MIN_COUNT:-3}"
BATCH_SEQS="${BATCH_SEQS:-32}"
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"

case "$CANDIDATE" in
  record659)
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659.txt}"
    ;;
  record659_smoke)
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_smoke.txt}"
    ;;
  lowrisk)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk.txt}"
    ;;
  lowrisk_smoke)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk_smoke.txt}"
    ;;
  lam10_conf05)
    NGRAM_LAMBDA="0.10"
    CONFIDENCE_THRESHOLD="0.5"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lam10_conf05.txt}"
    ;;
  vr1_record659)
    VALUE_RESIDUAL="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_vr1_record659.txt}"
    ;;
  *)
    echo "unknown artifact ngram candidate: $CANDIDATE" >&2
    exit 1
    ;;
esac

rm -f "$LOG_PATH"

exec python scripts/eval_ngram_cache_artifact.py \
  --run-dir "$RUN_DIR" \
  --log-path "$LOG_PATH" \
  --batch-seqs "$BATCH_SEQS" \
  --bigram-vocab-size "$BIGRAM_VOCAB_SIZE" \
  --value-residual "$VALUE_RESIDUAL" \
  --stride "$STRIDE" \
  --ngram-lambda "$NGRAM_LAMBDA" \
  --ngram-max-n "$NGRAM_MAX_N" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD" \
  --min-count "$MIN_COUNT" \
  --max-windows "$MAX_WINDOWS"
