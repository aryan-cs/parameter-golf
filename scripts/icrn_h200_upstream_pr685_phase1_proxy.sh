#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr685-worktree}"
PR685_RECORD_DIR="records/track_10min_16mb/2026-03-25_ChainedTTT_8xH100"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-1337}"
RUN_ID="${RUN_ID:-h200_upstream_pr685_phase1_proxy7185_seed${SEED}}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
ITERATIONS="${ITERATIONS:-7185}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr685
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

cd "$WORKTREE_DIR/$PR685_RECORD_DIR"

env \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  ITERATIONS="$ITERATIONS" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  TTT_EPOCHS="${TTT_EPOCHS:-20}" \
  TTT_LR="${TTT_LR:-0.0005}" \
  TTT_PASSES="${TTT_PASSES:-1}" \
  /usr/bin/time -f 'elapsed=%E' \
  torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_PATH"
