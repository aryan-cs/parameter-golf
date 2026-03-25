#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_candidate() {
  local candidate="$1"
  case "$candidate" in
    baseline)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon.sh"
      ;;
    upstream_pr674_exact)
      exec bash "$ROOT_DIR/scripts/h100_upstream_pr674_exact.sh"
      ;;
    upstream_pr676_exact)
      exec bash "$ROOT_DIR/scripts/h100_upstream_pr676_exact.sh"
      ;;
    upstream_pr685_phase1_exact)
      exec bash "$ROOT_DIR/scripts/h100_upstream_pr685_phase1_exact.sh"
      ;;
    upstream_pr684_exact)
      exec bash "$ROOT_DIR/scripts/h100_upstream_pr684_exact.sh"
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
    swiglu_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" swiglu ngram674
      ;;
    swiglu676_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" swiglu676 ngram674
      ;;
    swiglu_ngram659_conf07)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" swiglu ngram659_conf07
      ;;
    rope24_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" rope24 ngram674
      ;;
    rope24_ngram659_conf07)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" rope24 ngram659_conf07
      ;;
    xsa11_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" xsa11 ngram674
      ;;
    xsa11_ngram659_conf07)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" xsa11 ngram659_conf07
      ;;
    podracing674_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674 ngram674
      ;;
    podracing674_swiglu_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674_swiglu ngram674
      ;;
    podracing674_xsa11_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674_xsa11 ngram674
      ;;
    warmup0)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0.sh"
      ;;
    warmup0_vr1_bg3072_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_tttlr25.sh"
      ;;
    warmup0_rope24_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" rope24 ngram674
      ;;
    warmup0_rope24_ngram659_conf07)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" rope24 ngram659_conf07
      ;;
    warmup0_xsa11_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" xsa11 ngram674
      ;;
    warmup0_xsa11_ngram659_conf07)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" xsa11 ngram659_conf07
      ;;
    warmup0_podracing674_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674 ngram674
      ;;
    warmup0_podracing674_swiglu_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674_swiglu ngram674
      ;;
    warmup0_swiglu676_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" swiglu676 ngram674
      ;;
    warmup0_podracing674_xsa11_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" podracing674_xsa11 ngram674
      ;;
    ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" baseline ngram674
      ;;
    ngram659_lamcool)
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_latecool_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_latecool_conf07_lamtail)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.72:0.12,0.80:0.09,0.90:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_latecool_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf07_lamcool)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_cool_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_cool_conf07_lamcool)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_cool_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf08)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.8}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf07_min5)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-5}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_conf07_lam20)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    ngram659_tgate40_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.4}"
      export NGRAM_GATE_MODE="${NGRAM_GATE_MODE:-target}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659.sh"
      ;;
    warmup0_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" baseline ngram674
      ;;
    warmup0_ngram659_lamcool)
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_latecool_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_latecool_conf07_lamtail)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.72:0.12,0.80:0.09,0.90:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_latecool_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_conf07_lamcool)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_cool_conf07)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_cool_conf07_lamcool)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      export NGRAM_LAMBDA_SCHEDULE="${NGRAM_LAMBDA_SCHEDULE:-0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_cool_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_conf08)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.8}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_conf07_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_conf07_lam20)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    warmup0_ngram659_tgate40_min4)
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.4}"
      export NGRAM_GATE_MODE="${NGRAM_GATE_MODE:-target}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659.sh"
      ;;
    vr1_bg3072_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_ngram659.sh"
      ;;
    vr1_bg3072_ngram674)
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" vr1_bg3072 ngram674
      ;;
    warmup0_vr1_bg3072_ngram659)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_ngram659.sh"
      ;;
    warmup0_vr1_bg3072_ngram674)
      export WARMUP_STEPS="${WARMUP_STEPS:-0}"
      exec bash "$ROOT_DIR/scripts/h100_record_push_candidate.sh" vr1_bg3072 ngram674
      ;;
    ngram659_adapt)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_adapt.sh"
      ;;
    ngram659_adapt_last2)
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-2}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_adapt.sh"
      ;;
    ngram659_adapt_last4)
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_adapt.sh"
      ;;
    warmup0_ngram659_adapt)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_adapt.sh"
      ;;
    warmup0_ngram659_adapt_last2)
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-2}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_adapt.sh"
      ;;
    warmup0_ngram659_adapt_last4)
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-4}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_adapt.sh"
      ;;
    vr1_bg3072_ngram659_adapt)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_vr1_bg3072_ngram659_adapt.sh"
      ;;
    warmup0_vr1_bg3072_ngram659_adapt)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_vr1_bg3072_ngram659_adapt.sh"
      ;;
    ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    ngram659_adamw30ep_cosine)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    ngram659_adamw30ep_cosine_latecool)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    ngram659_adamw30ep_cosine_lr3e4)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0003}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    ngram659_adamw12ep_cosine)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-12}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_tttlr25)
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_adamw30ep_cosine)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_adamw30ep_cosine_latecool)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_adamw30ep_cosine_lr3e4)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0003}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      exec bash "$ROOT_DIR/scripts/h100_repro_leaky_ttt_parallel_muon_warmup0_ngram659_tttlr25.sh"
      ;;
    warmup0_ngram659_adamw12ep_cosine)
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-12}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
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
  CANDIDATE=upstream_pr674_exact bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=upstream_pr676_exact bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=upstream_pr685_phase1_exact bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=bg3072 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=swiglu_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=swiglu676_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=swiglu_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=rope24_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=rope24_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=xsa11_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=xsa11_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=podracing674_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=podracing674_swiglu_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=podracing674_xsa11_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_rope24_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_rope24_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_xsa11_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_xsa11_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_podracing674_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_podracing674_swiglu_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_swiglu676_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_podracing674_xsa11_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_latecool_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_latecool_conf07_lamtail bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_latecool_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf07_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_cool_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_cool_conf07_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_cool_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf08 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf07_min5 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_conf07_lam20 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_tgate40_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_latecool_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_latecool_conf07_lamtail bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_latecool_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_conf07_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_cool_conf07 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_cool_conf07_lamcool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_cool_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_conf08 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_conf07_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_conf07_lam20 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_tgate40_min4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram659 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram674 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adapt bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adapt_last2 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adapt_last4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adapt bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adapt_last2 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adapt_last4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram659_adapt bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram659_adapt bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adamw30ep_cosine bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adamw30ep_cosine_latecool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adamw30ep_cosine_lr3e4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=ngram659_adamw12ep_cosine bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adamw30ep_cosine bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adamw30ep_cosine_latecool bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adamw30ep_cosine_lr3e4 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_ngram659_adamw12ep_cosine bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=vr1_bg3072_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh
  CANDIDATE=warmup0_vr1_bg3072_ngram659_tttlr25 bash $ROOT_DIR/scripts/h100_parallel_candidate_portfolio.sh

Candidate meanings:
  baseline             recovered winning stack
  upstream_pr674_exact exact upstream PR #674 root-trainer frontier run via worktree
  upstream_pr676_exact exact upstream PR #676 record-folder SwiGLU run via worktree
  upstream_pr685_phase1_exact
                       exact upstream PR #685 record-folder code with TTT_PASSES=1 for a legal cosine-recovery hedge
  vr1                  baseline + VALUE_RESIDUAL=1
  bg3072               baseline + BIGRAM_VOCAB_SIZE=3072
  vr1_bg3072           baseline + VALUE_RESIDUAL=1 + BIGRAM_VOCAB_SIZE=3072
  tttlr25              baseline + TTT_LR=0.0025
  vr1_bg3072_tttlr25   combo bet on all three high-signal knobs
  swiglu_ngram674      SwiGLU architecture hedge plus PR #674 hashed eval semantics
  swiglu676_ngram674   exact PR #676 SwiGLU hedge plus PR #674 hashed eval semantics
  swiglu_ngram659_conf07
                       SwiGLU hedge plus the best completed PR #659-style conf07 exact-cache lane
  rope24_ngram674      deeper partial-RoPE hedge plus PR #674 hashed eval semantics
  rope24_ngram659_conf07
                       deeper partial-RoPE hedge plus the best completed PR #659-style conf07 lane
  xsa11_ngram674       all-layer XSA hedge plus PR #674 hashed eval semantics
  xsa11_ngram659_conf07
                       all-layer XSA hedge plus the best completed PR #659-style conf07 lane
  podracing674_ngram674
                       current best PR-#674-surrogate arch hedge plus hashed eval semantics
  podracing674_swiglu_ngram674
                       PR-#674 surrogate plus late-QAT-0.5 and parameter-neutral SwiGLU
  podracing674_xsa11_ngram674
                       PR-#674 surrogate plus all-layer XSA on top of hashed eval semantics
  warmup0              baseline + WARMUP_STEPS=0 to claw back timed-run headroom
  warmup0_vr1_bg3072_tttlr25
                       combo bet plus WARMUP_STEPS=0 for score and compliance
  warmup0_rope24_ngram674
                       rope24_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_rope24_ngram659_conf07
                       rope24_ngram659_conf07 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_xsa11_ngram674
                       xsa11_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_xsa11_ngram659_conf07
                       xsa11_ngram659_conf07 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_podracing674_ngram674
                       podracing674_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_podracing674_swiglu_ngram674
                       podracing674_swiglu_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_swiglu676_ngram674
                       swiglu676_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_podracing674_xsa11_ngram674
                       podracing674_xsa11_ngram674 plus WARMUP_STEPS=0 for more timing headroom
  ngram659             PR #659-style 5-gram eval cache with TTT disabled
  ngram674             PR #674-style hashed alpha=0.20, min_count=2, always-apply 5-gram mix
  ngram659_lamcool     PR #659 eval cache with lambda taper from 0.15 to 0.06 late
  ngram659_conf07      PR #659 eval cache with a stricter 0.7 confidence gate
  ngram659_latecool_conf07
                      hold conf=0.7 through the strong early/mid regime, then cool only in the final third
  ngram659_latecool_conf07_lamtail
                      late-only confidence cooldown plus a matching late lambda taper
  ngram659_latecool_conf07_min4
                      late-only confidence cooldown plus min_count=4
  ngram659_conf07_lamcool
                      conf=0.7 plus lambda taper from 0.15 to 0.06 late
  ngram659_cool_conf07 PR #659 eval cache with conf=0.7 early, then cooling to 0.55 late
  ngram659_cool_conf07_lamcool
                      cooldown confidence plus lambda taper
  ngram659_cool_conf07_min4
                      cool-down confidence schedule plus min_count=4
  ngram659_conf08      PR #659 eval cache with an even stricter 0.8 confidence gate
  ngram659_conf07_min4 PR #659 eval cache with conf=0.7 and min_count=4
  ngram659_conf07_min5 PR #659 eval cache with conf=0.7 and min_count=5
  ngram659_conf07_lam20
                       PR #659 eval cache with conf=0.7 and a stronger 0.20 n-gram mix
  ngram659_tgate40_min4
                       PR #659 eval cache with target-prob gate 0.4 and min_count=4
  warmup0_ngram659     ngram659 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram674     ngram674 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram659_conf07
                       ngram659_conf07 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram659_latecool_conf07
                       ngram659_latecool_conf07 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram659_latecool_conf07_lamtail
                       ngram659_latecool_conf07_lamtail plus WARMUP_STEPS=0 for timing headroom
  warmup0_ngram659_latecool_conf07_min4
                       ngram659_latecool_conf07_min4 plus WARMUP_STEPS=0 for timing headroom
  warmup0_ngram659_conf08
                       ngram659_conf08 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram659_conf07_min4
                       ngram659_conf07_min4 plus WARMUP_STEPS=0 for timing headroom
  warmup0_ngram659_conf07_lam20
                       ngram659_conf07_lam20 plus WARMUP_STEPS=0 for more timing headroom
  warmup0_ngram659_tgate40_min4
                       ngram659_tgate40_min4 plus WARMUP_STEPS=0 for timing headroom
  vr1_bg3072_ngram659  stronger architecture knobs plus PR #659 eval cache
  vr1_bg3072_ngram674  stronger architecture knobs plus PR #674 hashed eval semantics
  warmup0_vr1_bg3072_ngram659
                       architecture knobs plus PR #659 eval cache plus WARMUP_STEPS=0
  warmup0_vr1_bg3072_ngram674
                       PR #674 eval semantics plus architecture knobs plus WARMUP_STEPS=0
  ngram659_adapt       PR #659 eval cache plus RMSprop ngram adaptation
  ngram659_adapt_last2 PR #659 eval cache plus RMSprop adaptation on the last 2 blocks
  ngram659_adapt_last4 PR #659 eval cache plus RMSprop adaptation on the last 4 blocks
  warmup0_ngram659_adapt
                       ngram659_adapt plus WARMUP_STEPS=0 for timing headroom
  warmup0_ngram659_adapt_last2
                       last-2-block ngram adaptation plus WARMUP_STEPS=0
  warmup0_ngram659_adapt_last4
                       last-4-block ngram adaptation plus WARMUP_STEPS=0
  vr1_bg3072_ngram659_adapt
                       architecture knobs plus PR #659 eval cache plus RMSprop adaptation
  warmup0_vr1_bg3072_ngram659_adapt
                       strongest pure ngram-adapt bet with extra timing headroom
  ngram659_tttlr25     PR #659 eval cache combined with legal score-first TTT
  ngram659_adamw30ep_cosine
                       PR #672-style AdamW 30-epoch step-cosine grouped TTT on top of PR #659 eval cache
  ngram659_adamw30ep_cosine_latecool
                       PR #672-style cosine TTT plus late-only confidence cooldown
  ngram659_adamw30ep_cosine_lr3e4
                       same as above but with a gentler 3e-4 TTT LR
  ngram659_adamw12ep_cosine
                       shorter 12-epoch cosine-TTT hedge if 30 epochs overfits our artifact
  warmup0_ngram659_tttlr25
                       ngram659_tttlr25 plus WARMUP_STEPS=0 for timing headroom
  warmup0_ngram659_adamw30ep_cosine
                       PR #672-style cosine TTT plus WARMUP_STEPS=0
  warmup0_ngram659_adamw30ep_cosine_latecool
                       late-only cosine-TTT hedge plus WARMUP_STEPS=0
  warmup0_ngram659_adamw30ep_cosine_lr3e4
                       gentler 30-epoch cosine TTT plus WARMUP_STEPS=0
  warmup0_ngram659_adamw12ep_cosine
                       12-epoch cosine TTT hedge plus WARMUP_STEPS=0
  vr1_bg3072_ngram659_tttlr25
                       architecture knobs plus combined PR #659 eval cache and TTT
  warmup0_vr1_bg3072_ngram659_tttlr25
                       strongest combo bet with extra timing headroom

For a surviving candidate, rerun the significance seeds with:

  CANDIDATE=baseline bash $ROOT_DIR/scripts/h100_parallel_candidate_3seed.sh
EOF
