#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARCH_CANDIDATE="podracing674" \
exec bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_candidate.sh"
