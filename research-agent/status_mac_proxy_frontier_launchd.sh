#!/usr/bin/env bash
set -euo pipefail

LABEL="com.parameter-golf.mac-proxy-frontier"
if launchctl list | grep -q "$LABEL"; then
  launchctl list | grep "$LABEL"
  exit 0
fi

launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null || {
  echo "launchd service not loaded"
  exit 0
}
