#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <candidate> <seed> <source_run_name_prefix> <config_path> [poll_seconds]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$1"
SEED="$2"
SOURCE_PREFIX="$3"
CONFIG_PATH="$4"
POLL_SECONDS="${5:-30}"
LOG="${ROOT}/logs/${SOURCE_PREFIX}_export_queue_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${ROOT}/logs"

find_latest_run_dir() {
  local latest=""
  shopt -s nullglob
  for d in "${ROOT}/runs/${CANDIDATE}/seed${SEED}"/*; do
    [[ -d "$d" && -f "$d/config.env" ]] || continue
    if grep -q "^RUN_NAME_PREFIX=${SOURCE_PREFIX}$" "$d/config.env"; then
      latest="$d"
    fi
  done
  shopt -u nullglob
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

RUN_DIR=""
until RUN_DIR="$(find_latest_run_dir)"; [[ -n "$RUN_DIR" ]]; do
  printf '%s waiting for run dir for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  sleep "$POLL_SECONDS"
done

while pgrep -af "$SOURCE_PREFIX" >/dev/null 2>&1; do
  printf '%s source still running for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  sleep "$POLL_SECONDS"
done

CKPT_PATH="${RUN_DIR}/pre_export_model.pt"
until [[ -f "$CKPT_PATH" ]]; do
  printf '%s waiting for checkpoint %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CKPT_PATH" >> "$LOG"
  sleep "$POLL_SECONDS"
done

printf '%s launching export-only sweep from %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CKPT_PATH" >> "$LOG"
EXPORT_ONLY_CHECKPOINT="$CKPT_PATH" bash "${ROOT}/runpod/pod_run.sh" "$CANDIDATE" "$SEED" "$CONFIG_PATH" >> "$LOG" 2>&1
