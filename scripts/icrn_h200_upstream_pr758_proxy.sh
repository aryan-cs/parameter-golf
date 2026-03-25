#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr758-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_11L_XSA_7gram"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-1337}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
TIMED_MODE="${TIMED_MODE:-0}"

run_suffix=""
if [[ "$TIMED_MODE" == "1" ]]; then
  : "${WARMUP_STEPS:=0}"
  : "${VAL_LOSS_EVERY:=500}"
  : "${TRAIN_LOG_EVERY:=250}"
  : "${MAX_WALLCLOCK_SECONDS:=596}"
  run_suffix="_timed"
fi
if [[ "${COMPILE_ENABLED:-1}" == "0" ]]; then
  run_suffix="${run_suffix}_nocompile"
fi

RUN_ID="${RUN_ID:-h200_upstream_pr758_proxy600${run_suffix}_seed${SEED}}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr758-run-${RUN_ID}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr758; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/758/head:pr758
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr758
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

rm -rf "$PATCH_RUN_DIR"
mkdir -p "$PATCH_RUN_DIR"
cp -a --no-preserve=mode,ownership "$WORKTREE_DIR/." "$PATCH_RUN_DIR/"
python "$ROOT_DIR/scripts/patch_pr758_compile_gate.py" "$PATCH_RUN_DIR/$RECORD_DIR_REL/train_gpt.py"

cd "$ROOT_DIR"
source .venv/bin/activate

PATCH_RECORD_DIR="$PATCH_RUN_DIR/$RECORD_DIR_REL"
mkdir -p "$PATCH_RECORD_DIR/logs"
python -m py_compile "$PATCH_RECORD_DIR/train_gpt.py"
cd "$PATCH_RECORD_DIR"

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
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
  LEAKY_RELU="${LEAKY_RELU:-1}" \
  XSA_LAST_N="${XSA_LAST_N:-11}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-10240}" \
  TTT_ENABLED="${TTT_ENABLED:-0}" \
  NGRAM_CACHE="${NGRAM_CACHE:-1}" \
  NGRAM_ORDER="${NGRAM_ORDER:-7}" \
  NGRAM_ALPHA="${NGRAM_ALPHA:-0.40}" \
  NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-2}" \
  NGRAM_BUCKETS="${NGRAM_BUCKETS:-4194304}" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_PATH"
