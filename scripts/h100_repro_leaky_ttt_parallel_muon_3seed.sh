#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS="${SEEDS:-1337 42 2025}"
RUN_PREFIX="${RUN_PREFIX:-h100_repro_leaky_ttt_parallel_muon}"

for seed in $SEEDS; do
  echo "=== Starting seed ${seed} ==="
  SEED="$seed" RUN_ID="${RUN_PREFIX}_seed${seed}" \
    bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
done
