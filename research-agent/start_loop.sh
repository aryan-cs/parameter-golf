#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_ROOT="$SCRIPT_DIR"
cd "$AGENT_ROOT"

mkdir -p loop/runtime/logs

if [[ -f loop/runtime/controller.pid ]]; then
  pid="$(cat loop/runtime/controller.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    echo "loop already running with pid $pid"
    exit 0
  fi
fi

nohup python3 "$AGENT_ROOT/loopctl.py" start --config "$AGENT_ROOT/loop/config.json" > loop/runtime/logs/controller.stdout.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > loop/runtime/controller.pid
echo "started loop with pid $pid"
