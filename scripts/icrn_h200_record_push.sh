#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/record_push_candidate_lib.sh
source "$ROOT_DIR/scripts/record_push_candidate_lib.sh"

MODE="${1:-all}"
SEED="${SEED:-1337}"
FORCE="${FORCE:-0}"
STATUS_SCRIPT="$ROOT_DIR/scripts/record_push_status.py"

ARTIFACT_ORDER=(
  baseline
  tttlr25
  batch48
  tttlr25_batch48
  bg3072_tttlr25
  chunk16k
  epochs2_tttlr25
  freeze2_tttlr25
  freeze2_epochs2_tttlr25
  tttlr30
)

PROXY_ORDER=(
  baseline
  vr1
  bg3072
  vr1_bg3072
)

usage() {
  cat <<EOF
usage: $0 [artifact|proxy|combined|report|status|all|help]

Modes:
  artifact  Run the ordered artifact-only legal-TTT sweep.
  proxy     Run the ordered H200 H100-step proxy sweep.
  combined  Run the one combined proxy candidate, if both sweeps produced
            non-baseline winners.
  report    Print the ranked search-state report and exact 8xH100 handoff.
  status    Alias for report.
  all       Run artifact, proxy, combined, then report.
  help      Show this help text.

Environment:
  SEED   Seed used for proxy/combined runs. Default: 1337
  FORCE  Set to 1 to rerun completed candidates. Default: 0
EOF
}

log_has_metric() {
  local log_path="$1"
  local metric="$2"
  [[ -f "$log_path" ]] && rg -q "$metric" "$log_path"
}

run_artifact_stage() {
  local candidate log_path
  for candidate in "${ARTIFACT_ORDER[@]}"; do
    log_path="$(record_push_artifact_log_path "$ROOT_DIR" "$candidate")"
    if [[ "$FORCE" != "1" ]] && log_has_metric "$log_path" "legal_ttt_exact"; then
      echo "=== Skipping artifact candidate ${candidate}; found completed ${log_path} ==="
      continue
    fi
    echo "=== Running artifact candidate ${candidate} ==="
    CANDIDATE="$candidate" LOG_PATH="$log_path" \
      bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_candidate.sh"
  done
}

run_proxy_candidate() {
  local arch_candidate="$1"
  local ttt_candidate="${2:-baseline}"
  local label log_path
  label="$(record_push_candidate_slug "$arch_candidate" "$ttt_candidate")"
  log_path="$(record_push_proxy_log_path "$ROOT_DIR" "$arch_candidate" "$ttt_candidate" "$SEED")"
  if [[ "$FORCE" != "1" ]] && log_has_metric "$log_path" "legal_ttt_exact"; then
    echo "=== Skipping proxy candidate ${label}; found completed ${log_path} ==="
    return 0
  fi
  echo "=== Running proxy candidate ${label} (seed=${SEED}) ==="
  rm -f "$log_path"
  ARCH_CANDIDATE="$arch_candidate" TTT_CANDIDATE="$ttt_candidate" SEED="$SEED" \
    bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_candidate.sh" | tee "$log_path"
}

run_proxy_stage() {
  local arch_candidate
  for arch_candidate in "${PROXY_ORDER[@]}"; do
    run_proxy_candidate "$arch_candidate" "baseline"
  done
}

run_combined_stage() {
  local combined_json arch_candidate ttt_candidate completed
  combined_json="$(python "$STATUS_SCRIPT" --json --seed "$SEED")"
  readarray -t combined_info < <(
    python -c '
import json
import sys
d = json.loads(sys.stdin.read())
rec = d.get("recommended_combined")
if not rec:
    raise SystemExit(0)
print(rec["arch_candidate"])
print(rec["ttt_candidate"])
print("1" if rec.get("completed") else "0")
' <<<"$combined_json"
  )
  if [[ "${#combined_info[@]}" -eq 0 ]]; then
    echo "=== No combined candidate ready yet; need non-baseline winners from both sweeps ==="
    return 0
  fi
  arch_candidate="${combined_info[0]}"
  ttt_candidate="${combined_info[1]}"
  completed="${combined_info[2]}"
  if [[ "$FORCE" != "1" && "$completed" == "1" ]]; then
    echo "=== Skipping combined proxy candidate ${arch_candidate}+${ttt_candidate}; completed log already exists ==="
    return 0
  fi
  run_proxy_candidate "$arch_candidate" "$ttt_candidate"
}

print_report() {
  python "$STATUS_SCRIPT" --seed "$SEED"
}

case "$MODE" in
  artifact)
    run_artifact_stage
    ;;
  proxy)
    run_proxy_stage
    ;;
  combined)
    run_combined_stage
    ;;
  report)
    print_report
    ;;
  status)
    print_report
    ;;
  all)
    run_artifact_stage
    run_proxy_stage
    run_combined_stage
    print_report
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
