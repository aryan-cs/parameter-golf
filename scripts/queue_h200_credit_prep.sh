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

# 1b. Combine backward-looking n-gram scoring with legal TTT on the artifact.
CANDIDATE="${CANDIDATE_TTT_NGRAM_1:-record659_tttlr25_smoke}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"

# 1c. Cheap eval-side TTT search on the current best artifact.
CANDIDATE="${CANDIDATE_ARTIFACT_1:-tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"
CANDIDATE="${CANDIDATE_ARTIFACT_2:-tttlr25_batch48}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"
CANDIDATE="${CANDIDATE_ARTIFACT_3:-freeze2_epochs2_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_portfolio.sh"

# 1d. Full eval-only follow-ups after the smokes and cheap TTT variants.
CANDIDATE="${CANDIDATE_NGRAM_3:-record659}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_NGRAM_4:-lowrisk}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh"
CANDIDATE="${CANDIDATE_TTT_NGRAM_2:-record659_tttlr25}" \
  bash "$ROOT_DIR/scripts/icrn_h200_artifact_ttt_ngram_portfolio.sh"

# 2. Stricter H100-proxy search on the H200 with FA3 required.
RUN_ID="${RUN_ID_PROXY_1:-h200_ttt_h100proxy7185_warmup0_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_warmup0.sh"
RUN_ID="${RUN_ID_PROXY_2:-h200_ttt_h100proxy7185_vr1_bg3072_tttlr25_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_vr1_bg3072_tttlr25.sh"
RUN_ID="${RUN_ID_PROXY_3:-h200_ttt_h100proxy7185_warmup0_vr1_bg3072_tttlr25_seed1337}" \
  bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_warmup0_vr1_bg3072_tttlr25.sh"
