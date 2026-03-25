#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CANDIDATE:-}" ]]; then
  exec bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_candidate.sh"
fi

cat <<EOF
Artifact-only legal TTT candidate portfolio

Run one candidate at a time on the saved winning artifact by setting CANDIDATE:

  CANDIDATE=baseline bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh
  CANDIDATE=tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh
  CANDIDATE=tttlr30 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh
  CANDIDATE=batch48 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh
  CANDIDATE=chunk16k bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh
  CANDIDATE=bg3072_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh

Candidate meanings:
  baseline       recovered legal-TTT settings
  tttlr25        raise TTT_LR to 0.0025
  tttlr30        raise TTT_LR to 0.0030
  batch48        increase TTT_BATCH_SEQS to 48
  chunk16k       halve chunk size to 16384 tokens
  bg3072_tttlr25 raise BigramHash to 3072 and TTT_LR to 0.0025
EOF
