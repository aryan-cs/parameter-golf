#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr755-worktree}"
RECORD_DIR_REL="records/track_10min_16mb/2026-03-25_GravityTokenizer_AblationLeverage"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
MAX_SHARDS="${MAX_SHARDS:-0}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
INCLUDE_VAL="${INCLUDE_VAL:-1}"

STOCK_DIR="${STOCK_DIR:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
STOCK_TOKENIZER_PATH="${STOCK_TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
GRAVITY_TOKENIZER_PATH="${GRAVITY_TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/gravity_beta_1.0.model}"
if [[ "$MAX_SHARDS" == "0" && "$SHARD_OFFSET" == "0" ]]; then
  GRAVITY_DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0"
  setup_suffix="full"
else
  if [[ "$MAX_SHARDS" == "0" ]]; then
    setup_count="tail"
  else
    setup_count="$MAX_SHARDS"
  fi
  GRAVITY_DATA_PATH_DEFAULT="$ROOT_DIR/data/datasets/fineweb_gravity_beta_1.0_smoke${setup_count}"
  setup_suffix="offset${SHARD_OFFSET}_count${setup_count}"
fi
GRAVITY_DATA_PATH="${GRAVITY_DATA_PATH:-$GRAVITY_DATA_PATH_DEFAULT}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_upstream_pr755_setup_${setup_suffix}.txt}"

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT_DIR/data/tokenizers"
mkdir -p "$ROOT_DIR/data/datasets"

shard_is_complete_file() {
  local path="${1:?path required}"
  python3 - "$path" <<'PY'
import os
import sys
import numpy as np

path = sys.argv[1]
try:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError("bad header")
    num_tokens = int(header[2])
    expected_size = 256 * np.dtype("<i4").itemsize + num_tokens * np.dtype("<u2").itemsize
    ok = os.path.getsize(path) == expected_size
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

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
python "$ROOT_DIR/scripts/patch_pr755_retokenize_resume.py" "$PATCH_RECORD_DIR/retokenize_corpus.py"

if [[ ! -f "$STOCK_TOKENIZER_PATH" ]] || [[ ! -d "$STOCK_DIR" ]] || ! ls "$STOCK_DIR"/fineweb_train_*.bin >/dev/null 2>&1; then
  python3 data/cached_challenge_fineweb.py --variant sp1024
fi

cp "$PATCH_RECORD_DIR/gravity_beta_1.0.model" "$GRAVITY_TOKENIZER_PATH"

mapfile -t all_train_shards < <(find "$STOCK_DIR" -maxdepth 1 -name 'fineweb_train_*.bin' | sort)
mapfile -t all_val_shards < <(find "$STOCK_DIR" -maxdepth 1 -name 'fineweb_val_*.bin' | sort)
if [[ "${#all_train_shards[@]}" -eq 0 ]]; then
  echo "no training shards found under $STOCK_DIR" >&2
  exit 1
fi
if [[ "$SHARD_OFFSET" -lt 0 ]]; then
  echo "SHARD_OFFSET must be >= 0" >&2
  exit 1
fi
if [[ "$SHARD_OFFSET" -ge "${#all_train_shards[@]}" ]]; then
  echo "SHARD_OFFSET=$SHARD_OFFSET exceeds available train shard count ${#all_train_shards[@]}" >&2
  exit 1
fi

selected_train_shards=("${all_train_shards[@]:$SHARD_OFFSET}")
if [[ "$MAX_SHARDS" != "0" ]] && [[ "$MAX_SHARDS" -lt "${#selected_train_shards[@]}" ]]; then
  selected_train_shards=("${selected_train_shards[@]:0:$MAX_SHARDS}")
fi
expected_train_shards="${#selected_train_shards[@]}"
subset_stock_dir="$STOCK_DIR"
cleanup_subset_dir() {
  if [[ -n "${subset_stock_dir:-}" && "$subset_stock_dir" == "$ROOT_DIR"/data/datasets/.tmp_pr755_setup_* && -d "$subset_stock_dir" ]]; then
    rm -rf "$subset_stock_dir"
  fi
}
trap cleanup_subset_dir EXIT

if [[ "$SHARD_OFFSET" != "0" || "$MAX_SHARDS" != "0" || "$INCLUDE_VAL" != "1" ]]; then
  subset_stock_dir="$ROOT_DIR/data/datasets/.tmp_pr755_setup_offset${SHARD_OFFSET}_count${expected_train_shards}_val${INCLUDE_VAL}"
  rm -rf "$subset_stock_dir"
  mkdir -p "$subset_stock_dir"
  for shard_path in "${selected_train_shards[@]}"; do
    ln -s "$shard_path" "$subset_stock_dir/$(basename "$shard_path")"
  done
  if [[ "$INCLUDE_VAL" == "1" ]]; then
    for shard_path in "${all_val_shards[@]}"; do
      ln -s "$shard_path" "$subset_stock_dir/$(basename "$shard_path")"
    done
  fi
fi

existing_train_shards=0
existing_val_shards=0
if [[ -d "$GRAVITY_DATA_PATH" ]]; then
  for shard_path in "${selected_train_shards[@]}"; do
    output_path="$GRAVITY_DATA_PATH/$(basename "$shard_path")"
    if shard_is_complete_file "$output_path"; then
      existing_train_shards=$((existing_train_shards + 1))
    fi
  done
  if [[ "$INCLUDE_VAL" == "1" ]]; then
    for shard_path in "${all_val_shards[@]}"; do
      output_path="$GRAVITY_DATA_PATH/$(basename "$shard_path")"
      if shard_is_complete_file "$output_path"; then
        existing_val_shards=$((existing_val_shards + 1))
      fi
    done
  else
    existing_val_shards=1
  fi
fi

if [[ "$existing_train_shards" -ge "$expected_train_shards" && "$existing_val_shards" -ge 1 ]]; then
  echo "gravity dataset already present at $GRAVITY_DATA_PATH" | tee -a "$LOG_PATH"
  echo "existing_train_shards=${existing_train_shards} expected_train_shards=${expected_train_shards} existing_val_shards=${existing_val_shards}" | tee -a "$LOG_PATH"
  exit 0
fi

cmd=(
  python3
  "$PATCH_RECORD_DIR/retokenize_corpus.py"
  --base-tokenizer "$STOCK_TOKENIZER_PATH"
  --gravity-tokenizer "$GRAVITY_TOKENIZER_PATH"
  --data-dir "$subset_stock_dir"
  --output-dir "$GRAVITY_DATA_PATH"
)

printf 'gravity_setup:data=%s tokenizer=%s max_shards=%s shard_offset=%s include_val=%s source_data=%s expected_train_shards=%s\n' \
  "$GRAVITY_DATA_PATH" "$GRAVITY_TOKENIZER_PATH" "$MAX_SHARDS" "$SHARD_OFFSET" "$INCLUDE_VAL" "$subset_stock_dir" "$expected_train_shards" | tee "$LOG_PATH"
"${cmd[@]}" | tee -a "$LOG_PATH"
