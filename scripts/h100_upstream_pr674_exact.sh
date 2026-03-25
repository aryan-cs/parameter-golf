#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr674-worktree}"
SEED="${SEED:-2045}"
RUN_ID="${RUN_ID:-h100_upstream_pr674_seed${SEED}}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr674
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

cd "$WORKTREE_DIR"

env \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  MLP_ACT="${MLP_ACT:-leaky_relu_sq}" \
  MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.5}" \
  XSA_LAST_N="${XSA_LAST_N:-4}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
  ROPE_DIMS="${ROPE_DIMS:-24}" \
  NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-5}" \
  NGRAM_EVAL_ALPHA="${NGRAM_EVAL_ALPHA:-0.20}" \
  NGRAM_EVAL_MIN_COUNT="${NGRAM_EVAL_MIN_COUNT:-2}" \
  NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS:-4194304}" \
  NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS:-0.0}" \
  /usr/bin/time -f 'elapsed=%E' \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee "$LOG_PATH"
