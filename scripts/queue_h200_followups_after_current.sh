#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${CURRENT_PID:-35222}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

exec bash "$ROOT_DIR/scripts/queue_h200_credit_prep.sh"
