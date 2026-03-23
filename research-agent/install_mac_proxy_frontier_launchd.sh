#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="com.parameter-golf.mac-proxy-frontier"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
RUN_ID="${RUN_ID:-thwu1_mlx_mac_frontier_$(date +%Y%m%d_%H%M%S)}"
SEED_VALUE="${SEED:-42}"
RUN_DIR="$REPO_ROOT/research-experiments/runs/$RUN_ID"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$RUN_DIR"

cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO_ROOT/research-experiments/scripts/run_mac_proxy_frontier_when_idle.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$REPO_ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$RUN_DIR/launchd.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$RUN_DIR/launchd.stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/Users/aryan/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>RUN_ID</key>
    <string>$RUN_ID</string>
    <key>SEED</key>
    <string>$SEED_VALUE</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart "gui/$(id -u)/$LABEL"

echo "installed_launchd_label=$LABEL"
echo "plist=$PLIST"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
