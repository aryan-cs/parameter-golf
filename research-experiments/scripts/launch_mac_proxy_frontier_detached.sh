#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${RUN_ID:-thwu1_mlx_mac_frontier_$(date +%Y%m%d_%H%M%S)}"
SEED_VALUE="${SEED:-42}"
RUN_DIR="$REPO_ROOT/research-experiments/runs/$RUN_ID"
LOG_PATH="$RUN_DIR/launcher.log"
PID_PATH="$RUN_DIR/launcher.pid"

mkdir -p "$RUN_DIR"

nohup env RUN_ID="$RUN_ID" SEED="$SEED_VALUE" bash "$SCRIPT_DIR/run_mac_proxy_frontier_when_idle.sh" >"$LOG_PATH" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_PATH"

echo "launcher_pid=$pid"
echo "run_id=$RUN_ID"
echo "log_path=$LOG_PATH"
