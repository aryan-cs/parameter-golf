#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 <user@host> <checkpoint_path> [remote_dir] [port]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="$1"
CKPT_PATH="$2"
REMOTE_DIR="${3:-/workspace/golf}"
PORT="${4:-}"

SSH_ARGS=()
if [[ -n "$PORT" ]]; then
  SSH_ARGS+=(-p "$PORT")
fi

bash "${ROOT}/runpod/local_sync_to_pod.sh" "$REMOTE" "$REMOTE_DIR" "$PORT"

ssh "${SSH_ARGS[@]}" "$REMOTE" "
  set -euo pipefail
  cd '$REMOTE_DIR'
  export VENV_DIR=/root/.venvs/golf-export
  bash runpod/pod_bootstrap_export_only.sh
  bash runpod/pod_run_existing_export_ladder.sh \
    non_ttt_vrl_gptq \
    1337 \
    '$CKPT_PATH' \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune05.env \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune08.env \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune11.env \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune14.env \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune17.env \
    configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune20.env
"
