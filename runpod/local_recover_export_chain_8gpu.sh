#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 <user@host> [remote_dir] [port] [train_shards]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="$1"
REMOTE_DIR="${2:-/workspace/golf}"
PORT="${3:-}"
TRAIN_SHARDS="${4:-80}"

SSH_ARGS=()
if [[ -n "$PORT" ]]; then
  SSH_ARGS+=(-p "$PORT")
fi

bash "${ROOT}/runpod/local_sync_to_pod.sh" "$REMOTE" "$REMOTE_DIR" "$PORT"

ssh "${SSH_ARGS[@]}" "$REMOTE" "
  set -euo pipefail
  cd '$REMOTE_DIR'
  TRAIN_SHARDS='$TRAIN_SHARDS' bash runpod/pod_bootstrap.sh
  bash runpod/pod_launch_export_chain.sh \
    non_ttt_vrl_gptq \
    1337 \
    configs/runpod/non_ttt_vrl_gptq_8gpu.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune05.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune08.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune11.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune14.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune17.env \
    configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune20.env
"
