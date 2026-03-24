#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${CURRENT_PID:-35222}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

RUN_ID="${RUN_ID_H100_PROXY:-h200_ttt_h100proxy7185_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy.sh"

RUN_ID="${RUN_ID_H100_PROXY_VR1:-h200_ttt_h100proxy7185_vr1_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_vr1.sh"

RUN_ID="${RUN_ID_H100_PROXY_BG3072:-h200_ttt_h100proxy7185_bg3072_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_bg3072.sh"

RUN_ID="${RUN_ID_H100_PROXY_VR1_BG3072:-h200_ttt_h100proxy7185_vr1_bg3072_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_vr1_bg3072.sh"
