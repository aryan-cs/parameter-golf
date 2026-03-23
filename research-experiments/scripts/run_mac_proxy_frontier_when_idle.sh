#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CANDIDATE_RUNNER_PATTERN="scripts/run_mlx_proxy_experiment.py --experiment-dir mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy"
SLEEP_SECONDS="${MAC_PROXY_IDLE_POLL_SECONDS:-60}"

while pgrep -f "$CANDIDATE_RUNNER_PATTERN" >/dev/null 2>&1; do
  echo "waiting_for_idle pattern=$CANDIDATE_RUNNER_PATTERN sleep_seconds=$SLEEP_SECONDS"
  sleep "$SLEEP_SECONDS"
done

exec bash "$SCRIPT_DIR/run_mac_proxy_frontier.sh"
