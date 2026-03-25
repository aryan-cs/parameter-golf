#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

# 1. Highest-upside eval-side search first: backward-looking n-gram cache.
CANDIDATE="${CANDIDATE_NGRAM_1:-record659_smoke}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_NGRAM_2:-lowrisk_smoke}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_NGRAM_3:-record659}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_NGRAM_4:-lowrisk}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_NGRAM_5:-lam10_conf05}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"

# 1b. Combine backward-looking n-gram scoring with legal TTT on the artifact.
CANDIDATE="${CANDIDATE_TTT_NGRAM_1:-record659_tttlr25_smoke}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_TTT_NGRAM_2:-lowrisk_tttlr25_smoke}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_TTT_NGRAM_3:-record659_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_TTT_NGRAM_4:-lowrisk_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_TTT_NGRAM_5:-vr1_record659_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"

# 1c. Remaining cheap eval-side TTT search on the same artifact.
CANDIDATE="${CANDIDATE_ARTIFACT_1:-tttlr25_batch48}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"
CANDIDATE="${CANDIDATE_ARTIFACT_2:-freeze2_epochs2_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"
CANDIDATE="${CANDIDATE_ARTIFACT_3:-bg3072_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"

# 2. Resume the encoded proxy search without reopening warmup0 on H200.
bash "$ROOT_DIR/scripts/icrn_h200_record_push.sh" proxy
