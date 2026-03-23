#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CANDIDATE_ENTRY="$REPO_ROOT/research-experiments/mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy/train_gpt_mlx.py"
SLEEP_SECONDS="${MAC_PROXY_IDLE_POLL_SECONDS:-60}"

while pgrep -f "$CANDIDATE_ENTRY" >/dev/null 2>&1; do
  echo "waiting_for_idle candidate_entry=$CANDIDATE_ENTRY sleep_seconds=$SLEEP_SECONDS"
  sleep "$SLEEP_SECONDS"
done

exec bash "$SCRIPT_DIR/run_mac_proxy_frontier.sh"
