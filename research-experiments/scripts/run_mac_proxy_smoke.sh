#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv-mac/bin/python"
RUN_ID="${RUN_ID:-thwu1_mlx_mac_smoke_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "missing mac proxy environment at $VENV_PYTHON"
  echo "run: bash research-experiments/scripts/setup_mac_proxy_env.sh"
  exit 1
fi

cd "$REPO_ROOT/research-experiments"
uv run --python "$VENV_PYTHON" python scripts/run_mlx_proxy_experiment.py \
  --experiment-dir mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy \
  --run-id "$RUN_ID" \
  --seed "${SEED:-42}" \
  --set-env-file mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy/mac_smoke_env.json \
  --stats-path "runs/${RUN_ID}/stats.json"
