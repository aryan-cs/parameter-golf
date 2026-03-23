#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="$REPO_ROOT/research-experiments"
MANIFEST_DIR="$WORKDIR/manifests/pending"
RUN_ID="${RUN_ID:-thwu1_mlx_mac_frontier_$(date +%Y%m%d_%H%M%S)}"
SEED_VALUE="${SEED:-42}"
MANIFEST_PATH="$MANIFEST_DIR/${RUN_ID}.json"

mkdir -p "$MANIFEST_DIR"

cat >"$MANIFEST_PATH" <<EOF
{
  "id": "$RUN_ID",
  "job_kind": "proxy",
  "description": "Mac MLX frontier proxy run for the thwu1-derived candidate",
  "cwd": ".",
  "stats_path": "runs/$RUN_ID/stats.json",
  "required_paths": [
    "cache/openai-parameter-golf/data/datasets/fineweb10B_sp1024",
    "cache/openai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    "../.venv-mac/bin/python"
  ],
  "env": {
    "RUN_ID": "$RUN_ID",
    "SEED": "$SEED_VALUE"
  },
  "command": "bash scripts/run_mac_proxy_frontier.sh",
  "timeout_seconds": 32400
}
EOF

echo "staged_manifest=$MANIFEST_PATH"
echo "run_id=$RUN_ID"
