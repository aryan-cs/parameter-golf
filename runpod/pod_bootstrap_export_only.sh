#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-/root/.venvs/golf-export}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/logs"
LOG_PATH="${LOG_DIR}/bootstrap_export_${STAMP}.txt"

mkdir -p "$LOG_DIR" "$(dirname "$VENV_DIR")" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

PY="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

MISSING="$("$PY" - <<'PY'
import importlib.util
mods = {
    "numpy": "numpy",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "zstandard": "zstandard",
}
missing = [pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None]
print(" ".join(missing))
PY
)"

if [[ -n "$MISSING" ]]; then
  "$PIP" install --no-input $MISSING
fi

{
  echo "repo_root=$ROOT"
  echo "venv_dir=$VENV_DIR"
  echo "base_python=$("$BASE_PYTHON" --version 2>&1)"
  echo "venv_python=$("$PY" --version 2>&1)"
  echo "hf_home=$HF_HOME"
  echo "hf_hub_cache=$HUGGINGFACE_HUB_CACHE"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sync_commit=$(cat "$ROOT/.sync_commit" 2>/dev/null || echo unknown)"
  "$PY" - <<'PY'
import numpy, sentencepiece, torch
print(f"numpy={numpy.__version__}")
print(f"sentencepiece={sentencepiece.__version__}")
print(f"torch={torch.__version__}")
try:
    import zstandard
    print(f"zstandard={zstandard.__version__}")
except Exception:
    print("zstandard=missing")
PY
} | tee "$LOG_PATH"

echo "Export bootstrap complete."
