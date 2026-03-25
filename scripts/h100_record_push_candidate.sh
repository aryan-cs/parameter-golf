#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/record_push_candidate_lib.sh
source "$ROOT_DIR/scripts/record_push_candidate_lib.sh"

ARCH_CANDIDATE="${ARCH_CANDIDATE:-${1:-baseline}}"
TTT_CANDIDATE="${TTT_CANDIDATE:-${2:-baseline}}"
SEED="${SEED:-1337}"

record_push_apply_arch_candidate "$ARCH_CANDIDATE"
record_push_apply_ttt_candidate "$TTT_CANDIDATE"

slug="$(record_push_candidate_slug "$ARCH_CANDIDATE" "$TTT_CANDIDATE")"
if [[ "$slug" == "baseline" ]]; then
  export RUN_ID="${RUN_ID:-h100_record_push_baseline_seed${SEED}}"
else
  export RUN_ID="${RUN_ID:-h100_record_push_${slug}_seed${SEED}}"
fi
export SEED

exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
