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
  CANDIDATE=record659_lamcool bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_lamcool_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf06 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf06_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_hedge bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_hedge_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_hedge_eta05_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_hedge_eta20_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07_lamtail bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07_lamtail_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07_min4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_latecool_conf07_min4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_lamcool bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_lamcool_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07_lamcool bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07_lamcool_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07_min4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_cool_conf07_min4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf08 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf08_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_min4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_min4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_min5_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_conf07_min5 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_tgate30_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_tgate40_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_tgate40_min4_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_tgate40_min4 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_lam20_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674_hedge bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674_hedge_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674_proxy7185 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record674_hedge_proxy7185 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_orderlam bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_orderlam_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_orderlam bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_warm_conf07_orderlam_smoke bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
  CANDIDATE=record659_lam20_conf08 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
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
  record659_lamcool PR #659 confidence gate with lambda tapering from 0.15 to 0.06 late
  record659_lamcool_smoke lambda-cooldown variant on first 128 windows only
  record659_conf06 PR #659 lambda with confidence threshold raised to 0.6
  record659_conf06_smoke conf=0.6 on first 128 windows only
  record659_conf07 PR #659 lambda with confidence threshold raised to 0.7
  record659_conf07_smoke conf=0.7 on first 128 windows only
  record659_conf07_hedge conf=0.7 plus an online Hedge mixer between the base model and the PR #659 mixed expert
  record659_conf07_hedge_smoke Hedge mixer on first 128 windows only
  record659_conf07_hedge_eta05_smoke Hedge mixer smoke with slower adaptation (eta=0.05)
  record659_conf07_hedge_eta20_smoke Hedge mixer smoke with faster adaptation (eta=0.20)
  record659_latecool_conf07 hold conf=0.7 through the strong early/mid regime, then cool to 0.55 in the final third
  record659_latecool_conf07_smoke late-only cooldown variant on first 128 windows only
  record659_latecool_conf07_lamtail late-only confidence cooldown plus late lambda taper
  record659_latecool_conf07_lamtail_smoke late-only cooldown + lambda taper on first 128 windows only
  record659_latecool_conf07_min4 late-only cooldown plus min_count=4
  record659_latecool_conf07_min4_smoke late-only cooldown + min_count=4 on first 128 windows only
  record659_conf07_lamcool conf=0.7 plus lambda taper from 0.15 to 0.06 late
  record659_conf07_lamcool_smoke conf=0.7 plus lambda taper on first 128 windows only
  record659_cool_conf07 conf=0.7 early, then cool to 0.65/0.60/0.55 late
  record659_cool_conf07_smoke cooldown-confidence variant on first 128 windows only
  record659_cool_conf07_lamcool cooldown-confidence plus lambda taper
  record659_cool_conf07_lamcool_smoke cooldown-confidence plus lambda taper on first 128 windows only
  record659_cool_conf07_min4 cooldown-confidence variant with min_count=4
  record659_cool_conf07_min4_smoke cooldown-confidence + min_count=4 on first 128 windows only
  record659_conf08 PR #659 lambda with confidence threshold raised to 0.8
  record659_conf08_smoke conf=0.8 on first 128 windows only
  record659_conf07_min4_smoke conf=0.7 with min_count=4 on first 128 windows only
  record659_conf07_min4 conf=0.7 with min_count=4 full run
  record659_conf07_min5_smoke conf=0.7 with min_count=5 on first 128 windows only
  record659_conf07_min5 conf=0.7 with min_count=5 full run
  record659_tgate30_smoke target-prob gate at 0.3 on first 128 windows only
  record659_tgate40_smoke target-prob gate at 0.4 on first 128 windows only
  record659_tgate40_min4_smoke target-prob gate at 0.4 with min_count=4 on first 128 windows only
  record659_tgate40_min4 target-prob gate at 0.4 with min_count=4 full run
  record659_lam20_conf07 lambda=0.20 with conf=0.7 full run
  record674_smoke PR #674-inspired hashed score-first eval: alpha=0.20, min_count=2, 5-gram, 4M buckets on first 128 windows
  record674 PR #674-inspired hashed score-first eval: alpha=0.20, min_count=2, 5-gram, 4M buckets full run
  record674_hedge same hashed PR #674-inspired eval, but with an online Hedge mixer between neural and mixed experts
  record674_hedge_smoke hashed Hedge mixer on first 128 windows only
  record674_proxy7185 same hashed PR #674-inspired eval, intended for the saved proxy artifact via ARTIFACT_PATH/TEMPLATE_PATH
  record674_hedge_proxy7185 hashed Hedge mixer, intended for the saved proxy artifact via ARTIFACT_PATH/TEMPLATE_PATH
  record659_warm_conf07 staged confidence: 0.50 -> 0.60 -> 0.70 as cache warms up
  record659_warm_conf07_smoke staged-confidence variant on first 128 windows only
  record659_orderlam order-aware lambda ramp: 2:0.08, 3:0.12, 4:0.17, 5:0.22
  record659_orderlam_smoke order-aware lambda ramp on first 128 windows only
  record659_warm_conf07_orderlam staged confidence + order-aware lambda ramp
  record659_warm_conf07_orderlam_smoke combined staged-confidence/order-lambda variant on first 128 windows only
  record659_lam20_conf08 lambda=0.20 with conf=0.8 full run
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

Optional overrides:
  ARTIFACT_PATH=/abs/path/to/final_model.int6.ptz TEMPLATE_PATH=/abs/path/to/final_model.pt \
    CANDIDATE=record674_proxy7185 bash $ROOT_DIR/scripts/icrn_h200_artifact_ngram_portfolio.sh
EOF
