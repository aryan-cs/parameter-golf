#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
export RUN_ID="${RUN_ID:-h200_ttt_h100proxy7185_bg3072_seed${SEED:-1337}}"

exec bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy.sh"
