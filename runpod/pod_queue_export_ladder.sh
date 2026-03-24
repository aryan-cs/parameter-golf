#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <candidate> <seed> <source_run_name_prefix> <config_path> [config_path ...]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$1"
SEED="$2"
SOURCE_PREFIX="$3"
shift 3

resolve_run_name_prefix() {
  local config_path="$1"
  local run_name_prefix
  run_name_prefix="$(grep -E '^RUN_NAME_PREFIX=' "$config_path" | tail -n 1 | cut -d= -f2- || true)"
  if [[ -z "$run_name_prefix" ]]; then
    echo "missing RUN_NAME_PREFIX in $config_path" >&2
    exit 1
  fi
  printf '%s\n' "$run_name_prefix"
}

for config_path in "$@"; do
  run_name_prefix="$(resolve_run_name_prefix "$config_path")"
  nohup bash "${ROOT}/runpod/pod_queue_export_after_prefix.sh" \
    "$CANDIDATE" "$SEED" "$SOURCE_PREFIX" "$config_path" > /dev/null 2>&1 &
  pid="$!"
  printf 'queued %s after %s pid:%s\n' "$run_name_prefix" "$SOURCE_PREFIX" "$pid"
  SOURCE_PREFIX="$run_name_prefix"
done
