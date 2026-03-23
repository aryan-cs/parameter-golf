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

if [[ -n "$PORT" ]]; then
  RSYNC_ARGS+=(-e "ssh -p $PORT")
fi

mkdir -p "$ROOT/$SUBPATH"
rsync "${RSYNC_ARGS[@]}" "$REMOTE:$REMOTE_DIR/$SUBPATH/" "$ROOT/$SUBPATH/"

echo "Fetched $SUBPATH from $REMOTE:$REMOTE_DIR"
