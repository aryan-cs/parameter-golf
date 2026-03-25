#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr674-worktree}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr674-crownq-run}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-2045}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
ITERATIONS="${ITERATIONS:-7185}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-250}"
TIMED_MODE="${TIMED_MODE:-0}"

run_suffix=""
if [[ "$TIMED_MODE" == "1" ]]; then
  : "${WARMUP_STEPS:=0}"
  : "${VAL_LOSS_EVERY:=0}"
  : "${TRAIN_LOG_EVERY:=1000}"
  run_suffix="_timed"
fi
if [[ "${COMPILE_ENABLED:-1}" == "0" ]]; then
  run_suffix="${run_suffix}_nocompile"
fi

RUN_ID="${RUN_ID:-h200_upstream_pr674_crownq_proxy7185${run_suffix}_seed${SEED}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr674
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

rm -rf "$PATCH_RUN_DIR"
mkdir -p "$PATCH_RUN_DIR"
cp -a "$WORKTREE_DIR/." "$PATCH_RUN_DIR/"
python "$ROOT_DIR/scripts/patch_pr674_crownq.py" "$PATCH_RUN_DIR/train_gpt.py"

cd "$ROOT_DIR"
source .venv/bin/activate
python -m py_compile "$PATCH_RUN_DIR/train_gpt.py"
cd "$PATCH_RUN_DIR"

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
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
  COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
  MLP_ACT="${MLP_ACT:-leaky_relu_sq}" \
  MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.5}" \
  XSA_LAST_N="${XSA_LAST_N:-4}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
  ROPE_DIMS="${ROPE_DIMS:-24}" \
  LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.5}" \
  CROWNQ_LAMBDA="${CROWNQ_LAMBDA:-0.01}" \
  CROWNQ_WARMDOWN_ONLY="${CROWNQ_WARMDOWN_ONLY:-1}" \
  ITERATIONS="$ITERATIONS" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_PATH"
