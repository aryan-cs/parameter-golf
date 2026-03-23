#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CANDIDATE_RUNNER_PATTERN="scripts/run_mlx_proxy_experiment.py --experiment-dir mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy"
SLEEP_SECONDS="${MAC_PROXY_IDLE_POLL_SECONDS:-60}"

runner_is_active() {
  if pgrep -f "$CANDIDATE_RUNNER_PATTERN" >/dev/null 2>&1; then
    return 0
  fi
  local status=$?
  if [[ "$status" -eq 1 ]]; then
    return 1
  fi
  echo "idle_probe_unavailable pattern=$CANDIDATE_RUNNER_PATTERN pgrep_status=$status launching_without_wait"
  return 1
}

while runner_is_active; do
  echo "waiting_for_idle pattern=$CANDIDATE_RUNNER_PATTERN sleep_seconds=$SLEEP_SECONDS"
  sleep "$SLEEP_SECONDS"
done

exec bash "$SCRIPT_DIR/run_mac_proxy_frontier.sh"
