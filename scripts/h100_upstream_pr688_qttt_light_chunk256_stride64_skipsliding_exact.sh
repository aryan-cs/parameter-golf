#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  QTTT="${QTTT:-1}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-3}" \
  USE_POLYAK="${USE_POLYAK:-0}" \
  ADAPTIVE_LR="${ADAPTIVE_LR:-0}" \
  TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-262144}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  SKIP_SLIDING="${SKIP_SLIDING:-1}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr688_exact.sh"
