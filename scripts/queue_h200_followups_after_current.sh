#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${CURRENT_PID:-35222}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

RUN_ID="${RUN_ID_VR1:-h200_ttt_recordstack_vr1_80shard_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_value_residual.sh"

RUN_ID="${RUN_ID_BG3072:-h200_ttt_recordstack_bg3072_80shard_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_bigram3072.sh"
