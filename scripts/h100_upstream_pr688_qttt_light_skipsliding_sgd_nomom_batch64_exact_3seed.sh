#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  QTTT="${QTTT:-1}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-3}" \
  USE_POLYAK="${USE_POLYAK:-0}" \
  ADAPTIVE_LR="${ADAPTIVE_LR:-0}" \
  SKIP_SLIDING="${SKIP_SLIDING:-1}" \
  TTT_OPTIMIZER="${TTT_OPTIMIZER:-sgd}" \
  TTT_MOMENTUM="${TTT_MOMENTUM:-0.0}" \
  TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-64}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr688_exact_3seed.sh"
