#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 42 2045 7; do
  echo "=== PR753 exact seed ${seed} ==="
  SEED="$seed" bash "$ROOT_DIR/scripts/h100_upstream_pr753_exact.sh"
done
