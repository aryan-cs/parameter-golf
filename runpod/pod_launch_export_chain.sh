#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <candidate> <seed> <source_config_path> <export_config_path> [export_config_path ...]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$1"
SEED="$2"
SOURCE_CONFIG="$3"
shift 3

if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "missing source config: $SOURCE_CONFIG" >&2
  exit 1
fi

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

SOURCE_PREFIX="$(resolve_run_name_prefix "$SOURCE_CONFIG")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/logs"
LOG_PATH="${LOG_DIR}/${SOURCE_PREFIX}_launch_${STAMP}.log"
mkdir -p "$LOG_DIR"

nohup bash "${ROOT}/runpod/pod_run.sh" "$CANDIDATE" "$SEED" "$SOURCE_CONFIG" > "$LOG_PATH" 2>&1 &
RUN_PID="$!"
printf 'launched %s pid:%s log:%s\n' "$SOURCE_PREFIX" "$RUN_PID" "$LOG_PATH"

bash "${ROOT}/runpod/pod_queue_export_ladder.sh" "$CANDIDATE" "$SEED" "$SOURCE_PREFIX" "$@"
