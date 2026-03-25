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
  CANDIDATE=lowrisk bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lowrisk_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lam10_conf05 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=vr1_record659 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_adapt bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=lowrisk_adapt bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh

Candidate meanings:
  record659     PR #659 settings: stride=128, lambda=0.15, conf=0.5
  record659_smoke PR #659 settings on first 128 windows only
  lowrisk       gentler mix: lambda=0.05, conf=0.7
  lowrisk_smoke lowrisk settings on first 128 windows only
  lam10_conf05  middle mix: lambda=0.10, conf=0.5
  vr1_record659 PR #659 settings with VALUE_RESIDUAL=1 load path
  record659_adapt_smoke PR #659 settings plus RMSprop adapt on first 128 windows
  record659_adapt PR #659 settings plus RMSprop adapt on the full artifact
  lowrisk_adapt lowrisk n-gram mix plus RMSprop adapt
EOF
