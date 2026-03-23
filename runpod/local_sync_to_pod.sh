#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <user@host> [remote_dir]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
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
