#!/usr/bin/env bash
set -euo pipefail

export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
export TTT_LR="${TTT_LR:-0.0025}"
export RUN_ID="${RUN_ID:-h200_ttt_h100proxy7185_warmup0_vr1_bg3072_tttlr25_seed${SEED:-1337}}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/icrn_h200_ttt_h100_proxy.sh"
