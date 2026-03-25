#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CANDIDATE:-}" ]]; then
  exec bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh"
fi

cat <<EOF
Artifact-only backward-looking n-gram eval portfolio

Run one candidate at a time on the saved winning artifact by setting CANDIDATE:

  CANDIDATE=record659 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf06 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf06_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_orderlam bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_orderlam_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_orderlam bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_orderlam_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lowrisk bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lowrisk_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lam10_conf05 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=vr1_record659 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_last2_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_last2 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_last4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_last4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lowrisk_adapt bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh

Candidate meanings:
  All runs default to the packed-cache implementation for speed unless PACKED_CACHE=0.
  record659     PR #659 settings: stride=128, lambda=0.15, conf=0.5
  record659_smoke PR #659 settings on first 128 windows only
  record659_conf06 PR #659 lambda with confidence threshold raised to 0.6
  record659_conf06_smoke conf=0.6 on first 128 windows only
  record659_conf07 PR #659 lambda with confidence threshold raised to 0.7
  record659_conf07_smoke conf=0.7 on first 128 windows only
  record659_warm_conf07 staged confidence: 0.50 -> 0.60 -> 0.70 as cache warms up
  record659_warm_conf07_smoke staged-confidence variant on first 128 windows only
  record659_orderlam order-aware lambda ramp: 2:0.08, 3:0.12, 4:0.17, 5:0.22
  record659_orderlam_smoke order-aware lambda ramp on first 128 windows only
  record659_warm_conf07_orderlam staged confidence + order-aware lambda ramp
  record659_warm_conf07_orderlam_smoke combined staged-confidence/order-lambda variant on first 128 windows only
  lowrisk       gentler mix: lambda=0.05, conf=0.7
  lowrisk_smoke lowrisk settings on first 128 windows only
  lam10_conf05  middle mix: lambda=0.10, conf=0.5
  vr1_record659 PR #659 settings with VALUE_RESIDUAL=1 load path
  record659_adapt_smoke PR #659 settings plus RMSprop adapt on first 128 windows
  record659_adapt PR #659 settings plus RMSprop adapt on the full artifact
  record659_adapt_last2_smoke PR #659 settings plus RMSprop adapt on the last 2 blocks, first 128 windows
  record659_adapt_last2 PR #659 settings plus RMSprop adapt on the last 2 blocks
  record659_adapt_last4_smoke PR #659 settings plus RMSprop adapt on the last 4 blocks, first 128 windows
  record659_adapt_last4 PR #659 settings plus RMSprop adapt on the last 4 blocks
  lowrisk_adapt lowrisk n-gram mix plus RMSprop adapt
EOF
