#!/usr/bin/env bash

# Competition caps
export COMPETITION_ARTIFACT_LIMIT_BYTES="${COMPETITION_ARTIFACT_LIMIT_BYTES:-16000000}"
export COMPETITION_TRAIN_LIMIT_SECONDS="${COMPETITION_TRAIN_LIMIT_SECONDS:-600}"
export COMPETITION_EVAL_LIMIT_SECONDS="${COMPETITION_EVAL_LIMIT_SECONDS:-600}"

# Empirical 1xH200 proxy budget for a 10-minute 8xH100 training run on this stack.
# Anchors:
# - competition train cap: 600s on 8xH100
# - matched local H200 proxy baseline: 7185 steps in 4,615,816ms at ~642.42ms/step
# - implied 8xH100 reference step average from the cap: 600000 / 7185 ~= 83.51ms/step
export H100_PROXY_REFERENCE_STEPS="${H100_PROXY_REFERENCE_STEPS:-7185}"
export H100_PROXY_REFERENCE_STEP_AVG_MS="${H100_PROXY_REFERENCE_STEP_AVG_MS:-83.51}"
export H200_PROXY_STEP_AVG_MS="${H200_PROXY_STEP_AVG_MS:-642.42}"
export H200_PROXY_TRAIN_LIMIT_MS="${H200_PROXY_TRAIN_LIMIT_MS:-4615816}"
export H200_PROXY_TRAIN_LIMIT_SECONDS="${H200_PROXY_TRAIN_LIMIT_SECONDS:-4616}"
export H200_PROXY_TRAIN_LIMIT_MINUTES="${H200_PROXY_TRAIN_LIMIT_MINUTES:-76.9}"

h200_proxy_budget_note() {
  cat <<EOF
Competition caps:
  artifact <= ${COMPETITION_ARTIFACT_LIMIT_BYTES} bytes
  train <= ${COMPETITION_TRAIN_LIMIT_SECONDS}s on 8xH100
  eval <= ${COMPETITION_EVAL_LIMIT_SECONDS}s on 8xH100
Dev-side 1xH200 proxy for the train cap on this stack:
  <= ${H100_PROXY_REFERENCE_STEPS} steps
  <= ${H200_PROXY_TRAIN_LIMIT_MS}ms (~${H200_PROXY_TRAIN_LIMIT_MINUTES} min)
EOF
}

h200_proxy_guard_train_launch() {
  local iterations="${1:?iterations required}"
  local max_wallclock_seconds="${2:?max_wallclock_seconds required}"
  local allow_out_of_budget="${3:-0}"

  if [[ "$allow_out_of_budget" == "1" ]]; then
    return 0
  fi

  if (( iterations > H100_PROXY_REFERENCE_STEPS )); then
    echo "refusing out-of-budget H200 dev train launch: ITERATIONS=${iterations} exceeds proxy limit ${H100_PROXY_REFERENCE_STEPS}" >&2
    echo "set ALLOW_OUT_OF_BUDGET_DEV_RUN=1 only for an intentional unconstrained run" >&2
    return 1
  fi

  if [[ "$max_wallclock_seconds" != "0" ]]; then
    local rounded
    rounded="$(printf '%.0f' "$max_wallclock_seconds")"
    if (( rounded > H200_PROXY_TRAIN_LIMIT_SECONDS )); then
      echo "refusing out-of-budget H200 dev train launch: MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds} exceeds proxy limit ${H200_PROXY_TRAIN_LIMIT_SECONDS}" >&2
      echo "set ALLOW_OUT_OF_BUDGET_DEV_RUN=1 only for an intentional unconstrained run" >&2
      return 1
    fi
  fi
}
