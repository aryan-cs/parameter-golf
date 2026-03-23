#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT/research-experiments"

python3 scripts/run_record_experiment.py \
  --experiment-dir record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_colab_pilot_seed42 \
  --seed 42 \
  --nproc-per-node 1 \
  --required-cuda-devices 1 \
  --set-env-file record_candidates/2026-03-23_rank1_mixed_qat/colab_pilot_env.json \
  --stats-path runs/rank1_mixed_qat_colab_pilot_seed42/stats.json
