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
NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED:-0}"
NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR:-0.0003}"
NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY:-0.001}"
NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-3}"
CONFIDENCE_SCHEDULE="${CONFIDENCE_SCHEDULE:-}"
ORDER_LAMBDAS="${ORDER_LAMBDAS:-}"
PACKED_CACHE="${PACKED_CACHE:-1}"
BATCH_SEQS="${BATCH_SEQS:-32}"
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "$CANDIDATE" in
  record659)
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659.txt}"
    ;;
  record659_smoke)
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_smoke.txt}"
    ;;
  record659_conf06)
    CONFIDENCE_THRESHOLD="0.6"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf06.txt}"
    ;;
  record659_conf06_smoke)
    CONFIDENCE_THRESHOLD="0.6"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf06_smoke.txt}"
    ;;
  record659_conf07)
    CONFIDENCE_THRESHOLD="0.7"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07.txt}"
    ;;
  record659_conf07_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_smoke.txt}"
    ;;
  record659_warm_conf07)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07.txt}"
    ;;
  record659_warm_conf07_smoke)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_smoke.txt}"
    ;;
  record659_orderlam)
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_orderlam.txt}"
    ;;
  record659_orderlam_smoke)
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_orderlam_smoke.txt}"
    ;;
  record659_warm_conf07_orderlam)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_orderlam.txt}"
    ;;
  record659_warm_conf07_orderlam_smoke)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_orderlam_smoke.txt}"
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
  record659_adapt_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_smoke.txt}"
    ;;
  record659_adapt)
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt.txt}"
    ;;
  record659_adapt_last2_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="2"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last2_smoke.txt}"
    ;;
  record659_adapt_last2)
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="2"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last2.txt}"
    ;;
  record659_adapt_last4_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last4_smoke.txt}"
    ;;
  record659_adapt_last4)
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last4.txt}"
    ;;
  lowrisk_adapt)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk_adapt.txt}"
    ;;
  *)
    echo "unknown artifact ngram candidate: $CANDIDATE" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_COMPLETED" == "1" && -f "$LOG_PATH" ]] && rg -q "final_ngram_eval_exact" "$LOG_PATH"; then
  echo "skipping completed ngram candidate '$CANDIDATE' at $LOG_PATH"
  exit 0
fi

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
  --confidence-schedule "$CONFIDENCE_SCHEDULE" \
  --order-lambdas "$ORDER_LAMBDAS" \
  $( [[ "$NGRAM_ADAPT_ENABLED" == "1" ]] && printf '%s ' --ngram-adapt-enabled ) \
  $( [[ "$PACKED_CACHE" == "1" ]] && printf '%s ' --packed-cache ) \
  --ngram-adapt-lr "$NGRAM_ADAPT_LR" \
  --ngram-adapt-decay "$NGRAM_ADAPT_DECAY" \
  --ngram-adapt-last-n-blocks "$NGRAM_ADAPT_LAST_N_BLOCKS" \
  --max-windows "$MAX_WINDOWS"
