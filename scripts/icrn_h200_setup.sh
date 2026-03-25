#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# ICRN's H200 nodes currently expose a CUDA 12.8-capable driver, so we pin the
# matching PyTorch wheel instead of the default CUDA 13 build from PyPI.
python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128

python -m pip install einops
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  flash-attn-3

python - <<'PY'
import torch
from flash_attn_interface import flash_attn_func

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
print("flash_attn_3", "available")
PY
