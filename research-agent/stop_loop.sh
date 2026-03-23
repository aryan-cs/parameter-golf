#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_ROOT="$SCRIPT_DIR"
cd "$AGENT_ROOT"

if [[ ! -f loop/runtime/controller.pid ]]; then
  echo "no pid file found"
  exit 0
fi

pid="$(cat loop/runtime/controller.pid)"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "stopped loop pid $pid"
else
  echo "pid $pid is not running"
fi
rm -f loop/runtime/controller.pid
