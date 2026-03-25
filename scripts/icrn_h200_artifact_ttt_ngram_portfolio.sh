#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CANDIDATE:-}" ]]; then
  exec bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_candidate.sh"
fi

cat <<EOF
Artifact-only backward-looking n-gram + legal TTT portfolio

Run one candidate at a time on the saved winning artifact by setting CANDIDATE:

  CANDIDATE=record659_tttlr25_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=lowrisk_tttlr25_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=lowrisk_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=vr1_record659_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh

Candidate meanings:
  record659_tttlr25_smoke  PR #659 n-gram settings + TTT_LR=0.0025 on first 8 chunks
  record659_tttlr25        full PR #659 n-gram settings + TTT_LR=0.0025
  lowrisk_tttlr25_smoke    gentler n-gram mix + TTT_LR=0.0025 on first 8 chunks
  lowrisk_tttlr25          gentler n-gram mix + TTT_LR=0.0025 full run
  vr1_record659_tttlr25    PR #659 settings + VR1 load path + TTT_LR=0.0025
EOF
