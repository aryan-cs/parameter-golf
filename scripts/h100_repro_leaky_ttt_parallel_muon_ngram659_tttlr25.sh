#!/usr/bin/env bash
set -euo pipefail

export TTT_ENABLED="${TTT_ENABLED:-0}"
export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
export TTT_LR="${TTT_LR:-0.0025}"
export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
