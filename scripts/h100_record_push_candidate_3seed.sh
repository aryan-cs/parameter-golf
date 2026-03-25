#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="${SEEDS:-1337 42 2025}"
ARCH_CANDIDATE="${ARCH_CANDIDATE:-baseline}"
TTT_CANDIDATE="${TTT_CANDIDATE:-baseline}"

for seed in $SEEDS; do
  echo "=== Starting arch=${ARCH_CANDIDATE} ttt=${TTT_CANDIDATE} seed=${seed} ==="
  ARCH_CANDIDATE="$ARCH_CANDIDATE" TTT_CANDIDATE="$TTT_CANDIDATE" SEED="$seed" \
    bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh"
done
