#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 1337 42 7; do
  SEED="$seed" \
  RUN_ID="${RUN_ID:-h100_upstream_pr688_seed${seed}}" \
  bash "$ROOT_DIR/scripts/h100_upstream_pr688_exact.sh"
done
