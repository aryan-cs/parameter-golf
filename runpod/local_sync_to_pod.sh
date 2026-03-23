#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <user@host> [remote_dir] [port]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
PORT="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_ARGS=()
RSYNC_ARGS=(-az --delete)
if [[ -n "$PORT" ]]; then
  SSH_ARGS+=(-p "$PORT")
  RSYNC_ARGS+=(-e "ssh -p $PORT")
fi

ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync "${RSYNC_ARGS[@]}" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'data/datasets/' \
  --exclude 'data/tokenizers/' \
  --exclude 'logs/' \
  --exclude 'runs/' \
  "$ROOT/" "$REMOTE:$REMOTE_DIR/"

echo "Synced repo to $REMOTE:$REMOTE_DIR"
