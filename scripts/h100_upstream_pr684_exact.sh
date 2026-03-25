#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr684-worktree}"
PR684_RECORD_DIR="records/track_10min_16mb/2026-03-25_Sidecar48_Enhanced_Attention_Async_Data_Pipeline_AdamW_TTT"
SEED="${SEED:-1337}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
TIMED_MODE="${TIMED_MODE:-0}"

run_suffix=""
if [[ "$TIMED_MODE" == "1" ]]; then
  : "${WARMUP_STEPS:=0}"
  : "${VAL_LOSS_EVERY:=0}"
  : "${TRAIN_LOG_EVERY:=1000}"
  : "${MAX_WALLCLOCK_SECONDS:=596}"
  run_suffix="_timed"
fi
if [[ "${USE_COMPILE:-1}" == "0" ]]; then
  run_suffix="${run_suffix}_nocompile"
fi

RUN_ID="${RUN_ID:-h100_upstream_pr684${run_suffix}_seed${SEED}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr684
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

cd "$WORKTREE_DIR/$PR684_RECORD_DIR"

run_with_timer() {
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -f 'elapsed=%E' "$@"
  elif [[ -x /bin/time ]]; then
    /bin/time -f 'elapsed=%E' "$@"
  else
    TIMEFORMAT='elapsed=%R'
    time "$@"
  fi
}

run_with_timer env \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  WARMUP_STEPS="${WARMUP_STEPS:-20}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-596}" \
  USE_COMPILE="${USE_COMPILE:-1}" \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee "$LOG_PATH"
