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
LOG="${ROOT}/logs/${SOURCE_PREFIX}_export_ladder_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${ROOT}/logs"

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

find_latest_run_dir() {
  local prefix="$1"
  local latest=""
  shopt -s nullglob
  for d in "${ROOT}/runs/${CANDIDATE}/seed${SEED}"/*; do
    [[ -d "$d" && -f "$d/config.env" ]] || continue
    if grep -q "^RUN_NAME_PREFIX=${prefix}$" "$d/config.env"; then
      latest="$d"
    fi
  done
  shopt -u nullglob
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

run_is_size_ok() {
  local run_dir="$1"
  [[ -f "${run_dir}/train.log" ]] || return 1
  grep -Eq "Size OK:|size_ok:" "${run_dir}/train.log"
}

resolve_checkpoint_path() {
  local run_dir="$1"
  local ckpt="${run_dir}/pre_export_model.pt"
  if [[ -f "$ckpt" ]]; then
    printf '%s\n' "$ckpt"
    return 0
  fi
  if [[ -f "${run_dir}/env.txt" ]]; then
    local env_ckpt
    env_ckpt="$(grep -E '^EOC=' "${run_dir}/env.txt" | tail -n 1 | cut -d= -f2- || true)"
    if [[ -n "$env_ckpt" && -f "$env_ckpt" ]]; then
      printf '%s\n' "$env_ckpt"
      return 0
    fi
  fi
  return 1
}

SOURCE_RUN_DIR=""
until SOURCE_RUN_DIR="$(find_latest_run_dir "$SOURCE_PREFIX")"; [[ -n "$SOURCE_RUN_DIR" ]]; do
  printf '%s waiting for run dir for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  sleep 30
done

while pgrep -af "$SOURCE_PREFIX" >/dev/null 2>&1; do
  printf '%s source still running for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  sleep 30
done

if run_is_size_ok "$SOURCE_RUN_DIR"; then
  printf '%s source %s already produced valid-size artifact; stopping ladder\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  exit 0
fi

CKPT_PATH=""
until CKPT_PATH="$(resolve_checkpoint_path "$SOURCE_RUN_DIR")"; [[ -n "$CKPT_PATH" ]]; do
  printf '%s waiting for checkpoint for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_PREFIX" >> "$LOG"
  sleep 30
done

for config_path in "$@"; do
  run_name_prefix="$(resolve_run_name_prefix "$config_path")"
  printf '%s launching %s from %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_name_prefix" "$CKPT_PATH" >> "$LOG"
  EOC="$CKPT_PATH" bash "${ROOT}/runpod/pod_run.sh" "$CANDIDATE" "$SEED" "$config_path" >> "$LOG" 2>&1
  RUN_DIR=""
  until RUN_DIR="$(find_latest_run_dir "$run_name_prefix")"; [[ -n "$RUN_DIR" ]]; do
    printf '%s waiting for completed run dir for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_name_prefix" >> "$LOG"
    sleep 5
  done
  if run_is_size_ok "$RUN_DIR"; then
    printf '%s %s produced valid-size artifact; stopping ladder\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_name_prefix" >> "$LOG"
    exit 0
  fi
done

printf '%s export ladder exhausted without Size OK\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
