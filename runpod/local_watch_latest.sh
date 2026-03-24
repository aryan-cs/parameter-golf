#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo "usage: $0 <user@host> [remote_dir] [port] [candidate] [seed]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
PORT="${3:-}"
CANDIDATE="${4:-non_ttt_vrl_gptq}"
SEED="${5:-1337}"

SSH_ARGS=()
if [[ -n "$PORT" ]]; then
  SSH_ARGS+=(-p "$PORT")
fi

ssh "${SSH_ARGS[@]}" "$REMOTE" "
  set -euo pipefail
  cd '$REMOTE_DIR'
  latest=\$(find 'runs/$CANDIDATE/seed$SEED' -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
  if [[ -z \"\$latest\" ]]; then
    echo 'no run directory found for $CANDIDATE seed$SEED' >&2
    exit 1
  fi
  echo \"latest_run=\$latest\"
  tail -f \"\$latest/train.log\"
"
