#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_candidate() {
  local candidate="$1"
  case "$candidate" in
    baseline)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
      ;;
    vr1)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1.sh"
      ;;
    bg3072)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_bg3072.sh"
      ;;
    vr1_bg3072)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072.sh"
      ;;
    tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_tttlr25.sh"
      ;;
    vr1_bg3072_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_tttlr25.sh"
      ;;
    warmup0)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0.sh"
      ;;
    warmup0_vr1_bg3072_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_tttlr25.sh"
      ;;
    ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    warmup0_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    vr1_bg3072_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_ngram659.sh"
      ;;
    warmup0_vr1_bg3072_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_ngram659.sh"
      ;;
    ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_tttlr25.sh"
      ;;
    vr1_bg3072_ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_ngram659_tttlr25.sh"
      ;;
    warmup0_vr1_bg3072_ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_ngram659_tttlr25.sh"
      ;;
    *)
      echo "unknown candidate: $candidate" >&2
      exit 1
      ;;
  esac
}

if [[ -n "${CANDIDATE:-}" ]]; then
  run_candidate "$CANDIDATE"
fi

cat <<EOF
Parallel H100 candidate portfolio

Run one candidate on each 8xH100 node by setting CANDIDATE:

  CANDIDATE=baseline bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=bg3072 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh

Candidate meanings:
  baseline             recovered winning stack
  vr1                  baseline + VALUE_RESIDUAL=1
  bg3072               baseline + BIGRAM_VOCAB_SIZE=3072
  vr1_bg3072           baseline + VALUE_RESIDUAL=1 + BIGRAM_VOCAB_SIZE=3072
  tttlr25              baseline + TTT_LR=0.0025
  vr1_bg3072_tttlr25   combo bet on all three high-signal knobs
  warmup0              baseline + WARMUP_STEPS=0 to claw back timed-run headroom
  warmup0_vr1_bg3072_tttlr25
                       combo bet plus WARMUP_STEPS=0 for score and compliance
  ngram659             PR #659-style 5-gram eval cache with TTT disabled
  warmup0_ngram659     ngram659 plus WARMUP_STEPS=0 for more timing headroom
  vr1_bg3072_ngram659  stronger architecture knobs plus PR #659 eval cache
  warmup0_vr1_bg3072_ngram659
                       architecture knobs plus PR #659 eval cache plus WARMUP_STEPS=0
  ngram659_tttlr25     PR #659 eval cache combined with legal score-first TTT
  warmup0_ngram659_tttlr25
                       ngram659_tttlr25 plus WARMUP_STEPS=0 for timing headroom
  vr1_bg3072_ngram659_tttlr25
                       architecture knobs plus combined PR #659 eval cache and TTT
  warmup0_vr1_bg3072_ngram659_tttlr25
                       strongest combo bet with extra timing headroom

For a surviving candidate, rerun the significance seeds with:

  CANDIDATE=baseline bash $ROOT_DIR/scripts/h100_parallel_candidate_3seed.sh
EOF
