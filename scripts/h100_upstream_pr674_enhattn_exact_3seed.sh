#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 2045 1337 42; do
  echo "=== seed $seed ==="
  SEED="$seed" RUN_ID="h100_upstream_pr674_enhattn_seed${seed}" bash "$ROOT_DIR/scripts/h100_upstream_pr674_enhattn_exact.sh"
done
