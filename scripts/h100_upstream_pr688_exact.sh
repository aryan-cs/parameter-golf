#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr688-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-24_HedgeMixer_TTT"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
SEED="${SEED:-2045}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
RUN_ID="${RUN_ID:-h100_upstream_pr688_seed${SEED}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr688; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/688/head:pr688
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr688
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

RECORD_DIR="$WORKTREE_DIR/$RECORD_DIR_REL"
python -m py_compile "$RECORD_DIR/train_gpt.py"
cd "$RECORD_DIR"

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
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  USE_MIXER="${USE_MIXER:-1}" \
  MIXER_ETA="${MIXER_ETA:-0.1}" \
  TTT_LR="${TTT_LR:-0.0001}" \
  TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-131072}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}" \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee "$LOG_PATH"
