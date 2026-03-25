#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr755-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_GravityTokenizer_AblationLeverage"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-42}"
TIMED_MODE="${TIMED_MODE:-0}"
MAX_SHARDS="${MAX_SHARDS:-0}"

if [[ "$MAX_SHARDS" == "0" ]]; then
  DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0"
  run_suffix=""
else
  DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0_smoke${MAX_SHARDS}"
  run_suffix="_smoke${MAX_SHARDS}"
fi

DATA_PATH="${DATA_PATH:-$DATA_PATH_DEFAULT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/gravity_beta_1.0.model}"

if [[ "$TIMED_MODE" == "1" ]]; then
  : "${VAL_LOSS_EVERY:=0}"
  : "${TRAIN_LOG_EVERY:=1000}"
  : "${MAX_WALLCLOCK_SECONDS:=596}"
  run_suffix="${run_suffix}_timed"
fi

RUN_ID="${RUN_ID:-h200_upstream_pr755_proxy600${run_suffix}_seed${SEED}}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr755-run-${RUN_ID}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr755; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/755/head:pr755
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr755
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

if [[ ! -f "$TOKENIZER_PATH" ]] || [[ ! -d "$DATA_PATH" ]]; then
  echo "gravity dataset/tokenizer missing; run scripts/icrn_h200_upstream_pr755_setup.sh first" >&2
  exit 1
fi

rm -rf "$PATCH_RUN_DIR"
mkdir -p "$PATCH_RUN_DIR"
cp -a --no-preserve=mode,ownership "$WORKTREE_DIR/." "$PATCH_RUN_DIR/"

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
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}" \
  ITERATIONS="${ITERATIONS:-11000}" \
  WARMUP_STEPS="${WARMUP_STEPS:-50}" \
  WARMDOWN_ITERS="${WARMDOWN_ITERS:-2500}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  MODEL_DIM="${MODEL_DIM:-384}" \
  NUM_LAYERS="${NUM_LAYERS:-12}" \
  NUM_HEADS="${NUM_HEADS:-6}" \
  NUM_KV_HEADS="${NUM_KV_HEADS:-2}" \
  MLP_MULT="${MLP_MULT:-3}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}" \
  VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_PATH"
