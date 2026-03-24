#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 <pod_id> <checkpoint_path> [remote_dir] [poll_seconds]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_ID="$1"
CHECKPOINT_PATH="$2"
REMOTE_DIR="${3:-/workspace/golf}"
POLL_SECONDS="${4:-5}"

get_api_key() {
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    printf '%s\n' "$RUNPOD_API_KEY"
    return 0
  fi
  if [[ -f "${ROOT}/.env" ]]; then
    local key
    key="$(grep -E '^RUNPOD_API_KEY=' "${ROOT}/.env" | tail -n 1 | cut -d= -f2- || true)"
    if [[ -n "$key" ]]; then
      printf '%s\n' "$key"
      return 0
    fi
  fi
  echo "missing RUNPOD_API_KEY (env or ${ROOT}/.env)" >&2
  exit 1
}

API_KEY="$(get_api_key)"
API_URL="https://api.runpod.io/graphql"

run_gql() {
  local payload="$1"
  curl -fsS \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${API_KEY}" \
    --data "$payload" \
    "$API_URL"
}

resume_payload="$(python3 - <<'PY' "$POD_ID"
import json, sys
pod_id = sys.argv[1]
print(json.dumps({
    "query": f'mutation {{ podResume(input:{{podId:"{pod_id}"}}) {{ id desiredStatus }} }}'
}))
PY
)"

resume_json="$(run_gql "$resume_payload")"
resume_status="$(python3 - <<'PY' "$resume_json"
import json, sys
obj = json.loads(sys.argv[1])
if obj.get("errors"):
    err = obj["errors"][0]
    print(err.get("message", "unknown error"), file=sys.stderr)
    raise SystemExit(2)
pod = obj.get("data", {}).get("podResume")
if not pod:
    print("podResume returned no pod", file=sys.stderr)
    raise SystemExit(3)
print(pod.get("desiredStatus", "unknown"))
PY
)"

printf 'resume_status=%s\n' "$resume_status"

query_payload='{"query":"query { myself { pods { id desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}'

while true; do
  pod_json="$(run_gql "$query_payload")"
  line="$(python3 - <<'PY' "$pod_json" "$POD_ID"
import json, sys
obj = json.loads(sys.argv[1]); pod_id = sys.argv[2]
pods = obj.get("data", {}).get("myself", {}).get("pods", [])
for pod in pods:
    if pod.get("id") != pod_id:
        continue
    status = pod.get("desiredStatus", "")
    rt = pod.get("runtime") or {}
    ports = rt.get("ports") or []
    for port in ports:
        if port.get("isIpPublic") and port.get("type") == "tcp" and port.get("privatePort") == 22:
            print(f"{status} {port.get('ip')} {port.get('publicPort')}")
            raise SystemExit(0)
    print(status)
    raise SystemExit(0)
print("missing")
PY
)"
  read -r status host port <<<"$line"
  printf 'pod_status=%s\n' "$status"
  if [[ "$status" == "RUNNING" && -n "${host:-}" && -n "${port:-}" ]]; then
    printf 'ssh=%s:%s\n' "$host" "$port"
    bash "${ROOT}/runpod/local_recover_existing_export.sh" "root@${host}" "$CHECKPOINT_PATH" "$REMOTE_DIR" "$port"
    exit 0
  fi
  if [[ "$status" == "missing" ]]; then
    echo "pod not found: $POD_ID" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
