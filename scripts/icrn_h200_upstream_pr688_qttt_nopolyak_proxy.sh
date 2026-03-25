#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  QTTT="${QTTT:-1}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-4}" \
  USE_POLYAK="${USE_POLYAK:-0}" \
  bash "$ROOT_DIR/scripts/icrn_h200_upstream_pr688_proxy.sh"
