#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_LOG="${BASELINE_LOG:?set BASELINE_LOG}"
CANDIDATE_LOG="${CANDIDATE_LOG:?set CANDIDATE_LOG}"
NEXT_SCRIPT="${NEXT_SCRIPT:-}"
NEXT_LOG_PATH="${NEXT_LOG_PATH:-}"
NEXT_ENV_ASSIGNMENTS="${NEXT_ENV_ASSIGNMENTS:-}"
DECISION_STEP="${DECISION_STEP:-2500}"
MAX_DELTA_BPB="${MAX_DELTA_BPB:-0.0}"
MAX_STEP_AVG_RATIO="${MAX_STEP_AVG_RATIO:-1.20}"
CHECK_EVERY_SECONDS="${CHECK_EVERY_SECONDS:-60}"
KILL_PATTERN="${KILL_PATTERN:-torchrun --standalone --nproc_per_node=1 train_gpt.py}"
TEE_PATTERN="${TEE_PATTERN:-}"
DECISION_LOG="${DECISION_LOG:-/tmp/proxy_guard_decision.log}"

cd "$ROOT_DIR"

while [[ ! -f "$CANDIDATE_LOG" ]]; do
  sleep "$CHECK_EVERY_SECONDS"
done

decision_json() {
  python - "$BASELINE_LOG" "$CANDIDATE_LOG" "$DECISION_STEP" <<'PY'
import json
import re
import sys
from pathlib import Path

baseline_log = Path(sys.argv[1])
candidate_log = Path(sys.argv[2])
decision_step = int(sys.argv[3])
pat = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)\s+train_time:(\d+)ms step_avg:([0-9.]+)ms")

def parse(path):
    rows = {}
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            step = int(m.group(1))
            rows[step] = {
                "val_bpb": float(m.group(4)),
                "train_time_ms": int(m.group(5)),
                "step_avg_ms": float(m.group(6)),
            }
    return rows

b = parse(baseline_log)
c = parse(candidate_log)
common = sorted(set(b) & set(c))
eligible = [step for step in common if step >= decision_step]
if not eligible:
    print(json.dumps({"ready": False}))
    sys.exit(0)
step = eligible[0]
print(json.dumps({
    "ready": True,
    "step": step,
    "baseline_val_bpb": b[step]["val_bpb"],
    "candidate_val_bpb": c[step]["val_bpb"],
    "baseline_step_avg_ms": b[step]["step_avg_ms"],
    "candidate_step_avg_ms": c[step]["step_avg_ms"],
}))
PY
}

while true; do
  decision="$(decision_json)"
  ready="$(python - "$decision" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1]).get("ready") else "0")
PY
)"
  if [[ "$ready" == "1" ]]; then
    break
  fi
  sleep "$CHECK_EVERY_SECONDS"
done

kill_now="$(python - "$decision" "$MAX_DELTA_BPB" "$MAX_STEP_AVG_RATIO" "$DECISION_LOG" <<'PY'
import json
import sys
from pathlib import Path

decision = json.loads(sys.argv[1])
max_delta = float(sys.argv[2])
max_ratio = float(sys.argv[3])
decision_log = Path(sys.argv[4])

delta = decision["candidate_val_bpb"] - decision["baseline_val_bpb"]
ratio = decision["candidate_step_avg_ms"] / max(decision["baseline_step_avg_ms"], 1e-9)
kill = delta > max_delta or ratio > max_ratio

payload = dict(decision)
payload["delta_val_bpb"] = delta
payload["step_avg_ratio"] = ratio
payload["kill"] = kill
decision_log.write_text(json.dumps(payload, indent=2) + "\n")
print("1" if kill else "0")
PY
)"
if [[ "$kill_now" != "1" ]]; then
  exit 0
fi

pkill -f "$KILL_PATTERN" || true
if [[ -n "$TEE_PATTERN" ]]; then
  pkill -f "$TEE_PATTERN" || true
fi
sleep 5

if [[ -n "$NEXT_SCRIPT" ]]; then
  if [[ -n "$NEXT_LOG_PATH" ]] && [[ -f "$NEXT_LOG_PATH" ]]; then
    exit 0
  fi
  if [[ -n "$NEXT_ENV_ASSIGNMENTS" ]]; then
    read -r -a next_env <<< "$NEXT_ENV_ASSIGNMENTS"
    exec env "${next_env[@]}" bash "$NEXT_SCRIPT"
  fi
  exec bash "$NEXT_SCRIPT"
fi
