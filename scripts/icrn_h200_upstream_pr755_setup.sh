#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr755-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_GravityTokenizer_AblationLeverage"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
MAX_SHARDS="${MAX_SHARDS:-0}"

STOCK_DIR="${STOCK_DIR:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
STOCK_TOKENIZER_PATH="${STOCK_TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
GRAVITY_TOKENIZER_PATH="${GRAVITY_TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/gravity_beta_1.0.model}"
if [[ "$MAX_SHARDS" == "0" ]]; then
  GRAVITY_DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0"
  setup_suffix="full"
else
  GRAVITY_DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0_smoke${MAX_SHARDS}"
  setup_suffix="smoke${MAX_SHARDS}"
fi
GRAVITY_DATA_PATH="${GRAVITY_DATA_PATH:-$GRAVITY_DATA_PATH_DEFAULT}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_upstream_pr755_setup_${setup_suffix}.txt}"

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT_DIR/data/tokenizers"
mkdir -p "$ROOT_DIR/data/datasets"

if ! git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/pr755; then
  git -C "$ROOT_DIR" fetch https://github.com/openai/parameter-golf.git pull/755/head:pr755
fi
if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr755
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

PATCH_RECORD_DIR="$WORKTREE_DIR/$RECORD_DIR_REL"

if [[ ! -f "$STOCK_TOKENIZER_PATH" ]] || [[ ! -d "$STOCK_DIR" ]] || ! ls "$STOCK_DIR"/fineweb_train_*.bin >/dev/null 2>&1; then
  python3 data/cached_challenge_fineweb.py --variant sp1024
fi

cp "$PATCH_RECORD_DIR/gravity_beta_1.0.model" "$GRAVITY_TOKENIZER_PATH"

if [[ -d "$GRAVITY_DATA_PATH" ]] && ls "$GRAVITY_DATA_PATH"/fineweb_val_*.bin >/dev/null 2>&1; then
  if [[ "$MAX_SHARDS" == "0" ]] || ls "$GRAVITY_DATA_PATH"/fineweb_train_*.bin >/dev/null 2>&1; then
    echo "gravity dataset already present at $GRAVITY_DATA_PATH" | tee -a "$LOG_PATH"
    exit 0
  fi
fi

cmd=(
  python3
  "$PATCH_RECORD_DIR/retokenize_corpus.py"
  --base-tokenizer "$STOCK_TOKENIZER_PATH"
  --gravity-tokenizer "$GRAVITY_TOKENIZER_PATH"
  --data-dir "$STOCK_DIR"
  --output-dir "$GRAVITY_DATA_PATH"
)
if [[ "$MAX_SHARDS" != "0" ]]; then
  cmd+=(--max-shards "$MAX_SHARDS")
fi

printf 'gravity_setup:data=%s tokenizer=%s max_shards=%s\n' \
  "$GRAVITY_DATA_PATH" "$GRAVITY_TOKENIZER_PATH" "$MAX_SHARDS" | tee "$LOG_PATH"
"${cmd[@]}" | tee -a "$LOG_PATH"
