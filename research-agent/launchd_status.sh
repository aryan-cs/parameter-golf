#!/bin/zsh
set -euo pipefail

LABEL="com.parameter-golf.codex-loop"
launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null || {
  echo "launchd service not loaded"
  exit 0
}

