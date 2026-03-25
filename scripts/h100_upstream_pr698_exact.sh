#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr698-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_MergedTop3_v3_clean_h100"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-2045}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
TIMED_MODE="${TIMED_MODE:-0}"

run_suffix=""
if [[ "$TIMED_MODE" == "1" ]]; then
  : "${WARMUP_STEPS:=0}"
  : "${VAL_LOSS_EVERY:=0}"
  : "${TRAIN_LOG_EVERY:=1000}"
  : "${MAX_WALLCLOCK_SECONDS:=596}"
  run_suffix="_timed"
fi
if [[ "${COMPILE_ENABLED:-1}" == "0" ]]; then
  run_suffix="${run_suffix}_nocompile"
fi

RUN_ID="${RUN_ID:-h100_upstream_pr698${run_suffix}_seed${SEED}}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr698-run-${RUN_ID}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr698; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/698/head:pr698
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr698
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

rm -rf "$PATCH_RUN_DIR"
mkdir -p "$PATCH_RUN_DIR"
cp -a "$WORKTREE_DIR/." "$PATCH_RUN_DIR/"
python "$ROOT_DIR/scripts/patch_pr698_compile_gate.py" "$PATCH_RUN_DIR/$RECORD_DIR_REL/train_gpt.py"

cd "$ROOT_DIR"
source .venv/bin/activate

PATCH_RECORD_DIR="$PATCH_RUN_DIR/$RECORD_DIR_REL"
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
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee "$LOG_PATH"
