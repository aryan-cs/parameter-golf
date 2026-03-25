#!/usr/bin/env bash
set -euo pipefail

export WARMUP_STEPS="${WARMUP_STEPS:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_ngram659_adapt.sh"
