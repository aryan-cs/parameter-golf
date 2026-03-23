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
SYNC_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

SSH_ARGS=()
RSYNC_ARGS=(-az --delete)
TAR_EXCLUDES=(
  --exclude=.git
  --exclude=.venv
  --exclude=.env
  --exclude=__pycache__
  --exclude=.DS_Store
  --exclude=data/datasets
  --exclude=data/tokenizers
  --exclude=logs
  --exclude=runs
)
if [[ -n "$PORT" ]]; then
  SSH_ARGS+=(-p "$PORT")
  RSYNC_ARGS+=(-e "ssh -p $PORT")
fi

ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"
if ssh "${SSH_ARGS[@]}" "$REMOTE" "command -v rsync >/dev/null 2>&1"; then
  rsync "${RSYNC_ARGS[@]}" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude 'data/datasets/' \
    --exclude 'data/tokenizers/' \
    --exclude 'logs/' \
    --exclude 'runs/' \
    "$ROOT/" "$REMOTE:$REMOTE_DIR/"
else
  echo "Remote rsync not found; falling back to tar-over-SSH sync."
  ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"
  COPYFILE_DISABLE=1 tar -C "$ROOT" "${TAR_EXCLUDES[@]}" -cf - . | ssh "${SSH_ARGS[@]}" "$REMOTE" "tar --no-same-owner --no-same-permissions -C '$REMOTE_DIR' -xf -"
fi
ssh "${SSH_ARGS[@]}" "$REMOTE" "printf '%s\n' '$SYNC_COMMIT' > '$REMOTE_DIR/.sync_commit'"

echo "Synced repo to $REMOTE:$REMOTE_DIR"
