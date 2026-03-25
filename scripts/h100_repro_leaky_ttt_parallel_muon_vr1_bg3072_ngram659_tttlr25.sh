#!/usr/bin/env bash
set -euo pipefail

export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
