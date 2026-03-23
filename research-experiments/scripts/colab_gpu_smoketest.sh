#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT_ROOT="$REPO_ROOT/research-experiments"
cd "$EXPERIMENT_ROOT"

echo "repo_root=$REPO_ROOT"
echo "pwd=$(pwd)"
echo "hostname=$(hostname)"
uname -a

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "[nvidia-smi]"
  nvidia-smi
else
  echo
  echo "[nvidia-smi]"
  echo "nvidia-smi not found"
fi

echo
echo "[python runtime]"
python3 - <<'PY'
import importlib.util
import platform

print("platform", platform.platform())
for name in ("torch", "numpy", "sentencepiece", "zstandard"):
    print(f"module_{name}", bool(importlib.util.find_spec(name)))
try:
    import torch
    print("torch_cuda_available", torch.cuda.is_available())
    print("torch_cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"torch_cuda_name_{idx}", torch.cuda.get_device_name(idx))
except Exception as exc:
    print("torch_probe_error", repr(exc))
PY

echo
echo "[record preflight]"
set +e
python3 scripts/run_record_experiment.py \
  --experiment-dir record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_colab_smoketest_seed42 \
  --seed 42 \
  --nproc-per-node 1 \
  --required-cuda-devices 1 \
  --set-env-file record_candidates/2026-03-23_rank1_mixed_qat/colab_pilot_env.json \
  --stats-path runs/rank1_mixed_qat_colab_smoketest_seed42/stats.json \
  --preflight-only
status=$?
set -e

echo
echo "preflight_exit_code=$status"
echo "stats_path=$EXPERIMENT_ROOT/runs/rank1_mixed_qat_colab_smoketest_seed42/stats.json"
exit "$status"
