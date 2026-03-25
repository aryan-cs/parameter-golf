#!/usr/bin/env bash
set -euo pipefail

export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export RUN_ID="${RUN_ID:-h100_repro_leaky_ttt_parallel_muon_warmup0_seed${SEED:-1337}}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h100_repro_leaky_ttt_parallel_muon.sh"
