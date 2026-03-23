#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_ROOT="$EXPERIMENT_ROOT/cache/external"

mkdir -p "$EXTERNAL_ROOT"

sync_repo() {
  local url="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" fetch origin
    git -C "$target" reset --hard origin/HEAD
  else
    git clone --depth 1 "$url" "$target"
  fi
  git -C "$target" rev-parse HEAD
}

echo "syncing external public repos into $EXTERNAL_ROOT"
echo "thwu1 HEAD: $(sync_repo "https://github.com/thwu1/parameter-golf.git" "$EXTERNAL_ROOT/thwu1-parameter-golf")"
echo "aruniyer HEAD: $(sync_repo "https://github.com/aruniyer/parameter-golf.git" "$EXTERNAL_ROOT/aruniyer-parameter-golf")"
