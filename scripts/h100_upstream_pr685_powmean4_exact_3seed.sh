#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 1337 42 7; do
  echo "=== seed $seed ==="
  SEED="$seed" RUN_ID="h100_upstream_pr685_powmean4_seed${seed}" bash "$ROOT_DIR/scripts/h100_upstream_pr685_powmean4_exact.sh"
done
