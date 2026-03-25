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
  CANDIDATE=record659_late2_tttlr25_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw5e4_late2_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw1e4_late2_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_latecool_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_lamcool_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_lr3e4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw12ep_cosine_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_late2_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw5e4_late2 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw1e4_late2 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_latecool bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_lamcool bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=record659_adamw30ep_cosine_lr3e4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=lowrisk_tttlr25_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=lowrisk_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh
  CANDIDATE=vr1_record659_tttlr25 bash $ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh

Candidate meanings:
  record659_tttlr25_smoke  PR #659 n-gram settings + TTT_LR=0.0025 on first 8 chunks
  record659_late2_tttlr25_smoke  PR #659 settings + masked late-2-block SGD TTT_LR=0.0025 on first 8 chunks
  record659_adamw5e4_late2_smoke  PR #659 settings + masked late-2-block AdamW TTT_LR=5e-4 on first 8 chunks
  record659_adamw1e4_late2_smoke  PR #659 settings + masked late-2-block AdamW TTT_LR=1e-4 on first 8 chunks
  record659_adamw30ep_cosine_smoke  PR #672-style AdamW 30-epoch step-cosine grouped TTT on first 8 chunks
  record659_adamw30ep_cosine_latecool_smoke  PR #672-style grouped cosine TTT + late-only confidence cooldown on first 8 chunks
  record659_adamw30ep_cosine_lamcool_smoke  PR #672-style grouped cosine TTT + lambda taper on first 8 chunks
  record659_adamw30ep_cosine_lr3e4_smoke  same as above but lower TTT_LR=3e-4
  record659_adamw12ep_cosine_smoke  shorter 12-epoch PR #672-style grouped cosine TTT on first 8 chunks
  record659_tttlr25        full PR #659 n-gram settings + TTT_LR=0.0025
  record659_late2_tttlr25  full PR #659 settings + masked late-2-block SGD TTT_LR=0.0025
  record659_adamw5e4_late2 full PR #659 settings + masked late-2-block AdamW TTT_LR=5e-4
  record659_adamw1e4_late2 full PR #659 settings + masked late-2-block AdamW TTT_LR=1e-4
  record659_adamw30ep_cosine full PR #672-style AdamW 30-epoch step-cosine grouped TTT
  record659_adamw30ep_cosine_latecool full PR #672-style grouped cosine TTT + late-only confidence cooldown
  record659_adamw30ep_cosine_lamcool full PR #672-style grouped cosine TTT + lambda taper
  record659_adamw30ep_cosine_lr3e4 full PR #672-style grouped cosine TTT with TTT_LR=3e-4
  lowrisk_tttlr25_smoke    gentler n-gram mix + TTT_LR=0.0025 on first 8 chunks
  lowrisk_tttlr25          gentler n-gram mix + TTT_LR=0.0025 full run
  vr1_record659_tttlr25    PR #659 settings + VR1 load path + TTT_LR=0.0025
EOF
