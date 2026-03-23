#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_ROOT="$SCRIPT_DIR"
cd "$AGENT_ROOT"

python3 "$AGENT_ROOT/loopctl.py" status --config "$AGENT_ROOT/loop/config.json"
