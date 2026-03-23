#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_ROOT="$SCRIPT_DIR"
PYTHON="$(command -v python3)"
LABEL="com.parameter-golf.codex-loop"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$AGENT_ROOT/loop/runtime/logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>-u</string>
    <string>$AGENT_ROOT/loopctl.py</string>
    <string>start</string>
    <string>--config</string>
    <string>$AGENT_ROOT/loop/config.json</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$AGENT_ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>$AGENT_ROOT/loop/runtime/logs/launchd.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$AGENT_ROOT/loop/runtime/logs/launchd.stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/miniconda3/bin:/Users/aryan/.nvm/versions/node/v24.13.0/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart "gui/$(id -u)/$LABEL"

echo "installed and started $LABEL"
echo "plist: $PLIST"
