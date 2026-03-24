#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The public 8xH100 record stops around 7.18k steps under the 600s cap.
# On a single H200 we cannot use the same wallclock-based schedule directly,
# so proxy it with a matching step budget and step-based warmdown instead.
export ITERATIONS="${ITERATIONS:-7185}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export RUN_ID="${RUN_ID:-h200_ttt_h100proxy7185_seed${SEED:-1337}}"

exec bash "$ROOT_DIR/scripts/icrn_h200_ttt_recordstack.sh"
