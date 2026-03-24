#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <candidate> <seed> <checkpoint_path> <config_path> [config_path ...]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$1"
SEED="$2"
CKPT_PATH="$3"
shift 3

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "missing checkpoint: $CKPT_PATH" >&2
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

for config_path in "$@"; do
  if [[ ! -f "$config_path" ]]; then
    echo "missing config: $config_path" >&2
    exit 1
  fi
  run_name_prefix="$(resolve_run_name_prefix "$config_path")"
  echo "launching ${run_name_prefix} from ${CKPT_PATH}"
  EOC="$CKPT_PATH" bash "${ROOT}/runpod/pod_run.sh" "$CANDIDATE" "$SEED" "$config_path"
  run_dir="$(find_latest_run_dir "$run_name_prefix")"
  if [[ -z "$run_dir" ]]; then
    echo "missing run dir for ${run_name_prefix}" >&2
    exit 1
  fi
  if run_is_size_ok "$run_dir"; then
    echo "${run_name_prefix} produced valid-size artifact; stopping ladder"
    exit 0
  fi
done

echo "export ladder exhausted without Size OK"
