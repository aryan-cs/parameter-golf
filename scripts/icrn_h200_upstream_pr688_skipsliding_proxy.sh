#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  SKIP_SLIDING="${SKIP_SLIDING:-1}" \
  bash "$ROOT_DIR/scripts/icrn_h200_upstream_pr688_proxy.sh"
