#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_PATH="${LOG_PATH:-$RUN_DIR/logs/h200_ttt_recordstack_80shard_seed1337_resume_ttt.txt}"

cd "$ROOT_DIR"
source .venv/bin/activate

rm -f "$LOG_PATH"

exec python scripts/salvage_legal_ttt_eval.py \
  --run-dir "$RUN_DIR" \
  --log-path "$LOG_PATH" \
  --bigram-vocab-size 1536 \
  --ttt-freeze-blocks 0
