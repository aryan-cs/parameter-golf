#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr700-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_HedgeMixer_CROWNQ_Stride64_1.0541"
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

RUN_ID="${RUN_ID:-h200_upstream_pr700_proxy600${run_suffix}_seed${SEED}}"
PATCH_RUN_DIR="${PATCH_RUN_DIR:-$ROOT_DIR/../parameter-golf-pr700-run-${RUN_ID}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr700; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/700/head:pr700
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr700
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

rm -rf "$PATCH_RUN_DIR"
mkdir -p "$PATCH_RUN_DIR"
cp -a --no-preserve=mode,ownership "$WORKTREE_DIR/." "$PATCH_RUN_DIR/"
python "$ROOT_DIR/scripts/patch_pr700_compile_gate.py" "$PATCH_RUN_DIR/$RECORD_DIR_REL/train_gpt.py"

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
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
  USE_MIXER="${USE_MIXER:-1}" \
  MIXER_ETA="${MIXER_ETA:-0.1}" \
  TTT_EPOCHS="${TTT_EPOCHS:-4}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-2}" \
  TTT_LR="${TTT_LR:-0.0001}" \
  TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-131072}" \
  ADAPTIVE_LR="${ADAPTIVE_LR:-1}" \
  ADAPTIVE_LR_MAX="${ADAPTIVE_LR_MAX:-3.0}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  CROWN_Q_LAMBDA="${CROWN_Q_LAMBDA:-0.01}" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_PATH"
