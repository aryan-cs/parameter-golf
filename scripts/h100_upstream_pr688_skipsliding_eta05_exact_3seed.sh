#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  SKIP_SLIDING="${SKIP_SLIDING:-1}" \
  MIXER_ETA="${MIXER_ETA:-0.05}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr688_exact_3seed.sh"
