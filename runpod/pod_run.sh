#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <candidate_name> <seed> [config_path]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
VENV_DIR="${VENV_DIR:-/root/.venvs/golf}"
CANDIDATE="$1"
SEED="$2"
CONFIG_PATH="${3:-$ROOT/configs/runpod/${CANDIDATE}.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "missing config: $CONFIG_PATH" >&2
  exit 1
fi

set -a
source "$CONFIG_PATH"
set +a

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-${RUN_NAME_PREFIX:-$CANDIDATE}_s${SEED}_${STAMP}}"
RUN_DIR="$ROOT/runs/$CANDIDATE/seed${SEED}/$STAMP"
mkdir -p "$RUN_DIR"

if git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/commit.txt" 2>/dev/null; then
  :
elif [[ -f "$ROOT/.sync_commit" ]]; then
  cp "$ROOT/.sync_commit" "$RUN_DIR/commit.txt"
else
  printf 'unknown\n' > "$RUN_DIR/commit.txt"
fi
cp "$CONFIG_PATH" "$RUN_DIR/config.env"
cp "$ROOT/$TRAIN_SCRIPT" "$RUN_DIR/train_gpt.snapshot.py"
env | LC_ALL=C sort > "$RUN_DIR/env.txt"

export PYTHONUNBUFFERED=1
export UV_LINK_MODE=copy
export SEED
export RUN_ID
export OUT_DIR="$RUN_DIR"

cd "$RUN_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "missing virtualenv python at $VENV_PYTHON; run bash runpod/pod_bootstrap.sh first" >&2
  exit 1
fi

NPROC="${NPROC_PER_NODE:-1}"
if [[ "$NPROC" == "1" ]]; then
  CMD=("$VENV_PYTHON" "$ROOT/$TRAIN_SCRIPT")
else
  CMD=("$VENV_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$ROOT/$TRAIN_SCRIPT")
fi
printf 'command=%q ' "${CMD[@]}" | tee "$RUN_DIR/launch.txt"
printf '\n' | tee -a "$RUN_DIR/launch.txt"

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"

"$VENV_PYTHON" "$ROOT/runpod/collect_run_metadata.py" \
  --run-dir "$RUN_DIR" \
  --candidate "$CANDIDATE" \
  --seed "$SEED" \
  --train-script "$TRAIN_SCRIPT"

echo "Run complete: $RUN_DIR"
