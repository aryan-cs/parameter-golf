#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  SEED="${SEED:-2045}" \
  bash "$ROOT_DIR/scripts/icrn_h200_upstream_pr753_proxy.sh"
