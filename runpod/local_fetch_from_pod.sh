#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <user@host> [remote_dir] [subpath]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
SUBPATH="${3:-runs}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/$SUBPATH"
rsync -az "$REMOTE:$REMOTE_DIR/$SUBPATH/" "$ROOT/$SUBPATH/"

echo "Fetched $SUBPATH from $REMOTE:$REMOTE_DIR"
