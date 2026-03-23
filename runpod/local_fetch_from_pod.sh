#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 <user@host> [remote_dir] [subpath] [port]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
SUBPATH="${3:-runs}"
PORT="${4:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RSYNC_ARGS=(-az)
SSH_ARGS=()

if [[ -n "$PORT" ]]; then
  RSYNC_ARGS+=(-e "ssh -p $PORT")
  SSH_ARGS+=(-p "$PORT")
fi

mkdir -p "$ROOT/$SUBPATH"
if ssh "${SSH_ARGS[@]}" "$REMOTE" "command -v rsync >/dev/null 2>&1"; then
  rsync "${RSYNC_ARGS[@]}" "$REMOTE:$REMOTE_DIR/$SUBPATH/" "$ROOT/$SUBPATH/"
else
  echo "Remote rsync not found; falling back to tar-over-SSH fetch."
  ssh "${SSH_ARGS[@]}" "$REMOTE" "tar -C '$REMOTE_DIR' -cf - '$SUBPATH'" | tar -C "$ROOT" -xf -
fi

echo "Fetched $SUBPATH from $REMOTE:$REMOTE_DIR"
