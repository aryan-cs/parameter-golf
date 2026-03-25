#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
CANDIDATE="${CANDIDATE:-baseline}"

cd "$ROOT_DIR"
source .venv/bin/activate

BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"
TTT_LR="${TTT_LR:-0.002}"
TTT_EPOCHS="${TTT_EPOCHS:-3}"
TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "$CANDIDATE" in
  baseline)
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_baseline.txt}"
    ;;
  tttlr25)
    TTT_LR="0.0025"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_tttlr25.txt}"
    ;;
  tttlr30)
    TTT_LR="0.0030"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_tttlr30.txt}"
    ;;
  batch48)
    TTT_BATCH_SEQS="48"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_batch48.txt}"
    ;;
  tttlr25_batch48)
    TTT_LR="0.0025"
    TTT_BATCH_SEQS="48"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_tttlr25_batch48.txt}"
    ;;
  chunk16k)
    TTT_CHUNK_TOKENS="16384"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_chunk16k.txt}"
    ;;
  freeze2_tttlr25)
    TTT_FREEZE_BLOCKS="2"
    TTT_LR="0.0025"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_freeze2_tttlr25.txt}"
    ;;
  epochs2_tttlr25)
    TTT_EPOCHS="2"
    TTT_LR="0.0025"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_epochs2_tttlr25.txt}"
    ;;
  freeze2_epochs2_tttlr25)
    TTT_FREEZE_BLOCKS="2"
    TTT_EPOCHS="2"
    TTT_LR="0.0025"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_freeze2_epochs2_tttlr25.txt}"
    ;;
  bg3072_tttlr25)
    BIGRAM_VOCAB_SIZE="3072"
    TTT_LR="0.0025"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_bg3072_tttlr25.txt}"
    ;;
  *)
    echo "unknown artifact TTT candidate: $CANDIDATE" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_COMPLETED" == "1" && -f "$LOG_PATH" ]] && rg -q "legal_ttt_exact" "$LOG_PATH"; then
  echo "skipping completed artifact TTT candidate '$CANDIDATE' at $LOG_PATH"
  exit 0
fi

rm -f "$LOG_PATH"

exec python scripts/salvage_legal_ttt_eval.py \
  --run-dir "$RUN_DIR" \
  --log-path "$LOG_PATH" \
  --bigram-vocab-size "$BIGRAM_VOCAB_SIZE" \
  --ttt-freeze-blocks "$TTT_FREEZE_BLOCKS" \
  --ttt-lr "$TTT_LR" \
  --ttt-epochs "$TTT_EPOCHS" \
  --ttt-chunk-tokens "$TTT_CHUNK_TOKENS" \
  --ttt-momentum "$TTT_MOMENTUM" \
  --ttt-grad-clip "$TTT_GRAD_CLIP" \
  --batch-seqs "$TTT_BATCH_SEQS"
