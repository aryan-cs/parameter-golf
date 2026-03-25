#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
export TTT_LR="${TTT_LR:-0.0025}"
export RUN_ID="${RUN_ID:-h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_tttlr25_seed${SEED:-1337}}"

exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
