#!/usr/bin/env bash
set -euo pipefail

export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export RUN_ID="${RUN_ID:-h200_ttt_h100proxy7185_warmup0_seed${SEED:-1337}}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/icrn_h200_ttt_h100_proxy.sh"
