#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  QTTT="${QTTT:-1}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-3}" \
  USE_POLYAK="${USE_POLYAK:-0}" \
  ADAPTIVE_LR="${ADAPTIVE_LR:-0}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  bash "$ROOT_DIR/scripts/icrn_h200_upstream_pr688_proxy.sh"
