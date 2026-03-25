#!/usr/bin/env bash

record_push_apply_arch_candidate() {
  local candidate="${1:-baseline}"
  case "$candidate" in
    baseline|"")
      ;;
    vr1)
      export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
      ;;
    bg3072)
      export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
      ;;
    vr1_bg3072)
      export VALUE_RESIDUAL="${VALUE_RESIDUAL:-1}"
      export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
      ;;
    *)
      echo "unknown architecture candidate: $candidate" >&2
      return 1
      ;;
  esac
}

record_push_apply_ttt_candidate() {
  local candidate="${1:-baseline}"
  case "$candidate" in
    baseline|"")
      ;;
    tttlr25)
      export TTT_LR="${TTT_LR:-0.0025}"
      ;;
    tttlr30)
      export TTT_LR="${TTT_LR:-0.0030}"
      ;;
    batch48)
      export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-48}"
      ;;
    tttlr25_batch48)
      export TTT_LR="${TTT_LR:-0.0025}"
      export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-48}"
      ;;
    chunk16k)
      export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-16384}"
      ;;
    freeze2_tttlr25)
      export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-2}"
      export TTT_LR="${TTT_LR:-0.0025}"
      ;;
    epochs2_tttlr25)
      export TTT_EPOCHS="${TTT_EPOCHS:-2}"
      export TTT_LR="${TTT_LR:-0.0025}"
      ;;
    freeze2_epochs2_tttlr25)
      export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-2}"
      export TTT_EPOCHS="${TTT_EPOCHS:-2}"
      export TTT_LR="${TTT_LR:-0.0025}"
      ;;
    bg3072_tttlr25)
      export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
      export TTT_LR="${TTT_LR:-0.0025}"
      ;;
    ngram659)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_conf06)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.6}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_conf07)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_conf08)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.8}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_conf07_lam20)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_warm_conf07)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.50,0.20:0.60,0.40:0.70}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_orderlam)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_ORDER_LAMBDAS="${NGRAM_ORDER_LAMBDAS:-2:0.08,3:0.12,4:0.17,5:0.22}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_warm_conf07_orderlam)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_CONFIDENCE_SCHEDULE="${NGRAM_CONFIDENCE_SCHEDULE:-0.00:0.50,0.20:0.60,0.40:0.70}"
      export NGRAM_ORDER_LAMBDAS="${NGRAM_ORDER_LAMBDAS:-2:0.08,3:0.12,4:0.17,5:0.22}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    lowrisk_ngram)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.05}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    lam10_conf05_ngram)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.10}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_adapt)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED:-1}"
      export NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR:-0.0003}"
      export NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY:-0.001}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_adapt_last2)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      export NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED:-1}"
      export NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR:-0.0003}"
      export NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY:-0.001}"
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-2}"
      ;;
    ngram659_adapt_last4)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      export NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED:-1}"
      export NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR:-0.0003}"
      export NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY:-0.001}"
      export NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS:-4}"
      ;;
    lowrisk_ngram_adapt)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-1}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.05}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      export NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED:-1}"
      export NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR:-0.0003}"
      export NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY:-0.001}"
      ;;
    ngram659_tttlr25)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_LR="${TTT_LR:-0.0025}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      ;;
    ngram659_late2_tttlr25)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_LR="${TTT_LR:-0.0025}"
      export TTT_LAST_N_BLOCKS="${TTT_LAST_N_BLOCKS:-2}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      ;;
    ngram659_late2_adamw5e4)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_LAST_N_BLOCKS="${TTT_LAST_N_BLOCKS:-2}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      ;;
    ngram659_late2_adamw1e4)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0001}"
      export TTT_LAST_N_BLOCKS="${TTT_LAST_N_BLOCKS:-2}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_adamw30ep_cosine)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0005}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    ngram659_adamw30ep_cosine_lr3e4)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
      export TTT_LR="${TTT_LR:-0.0003}"
      export TTT_EPOCHS="${TTT_EPOCHS:-30}"
      export TTT_SCHEDULE="${TTT_SCHEDULE:-step_cosine}"
      export TTT_LR_GROUPING="${TTT_LR_GROUPING:-pr672}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.5}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    lowrisk_ngram_tttlr25)
      export TTT_ENABLED="${TTT_ENABLED:-0}"
      export NGRAM_EVAL_ENABLED="${NGRAM_EVAL_ENABLED:-0}"
      export NGRAM_TTT_ENABLED="${NGRAM_TTT_ENABLED:-1}"
      export TTT_LR="${TTT_LR:-0.0025}"
      export NGRAM_TTT_STRIDE="${NGRAM_TTT_STRIDE:-64}"
      export NGRAM_STRIDE="${NGRAM_STRIDE:-128}"
      export NGRAM_BATCH_SEQS="${NGRAM_BATCH_SEQS:-32}"
      export NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.05}"
      export NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
      export NGRAM_CONFIDENCE_THRESHOLD="${NGRAM_CONFIDENCE_THRESHOLD:-0.7}"
      export NGRAM_MIN_COUNT="${NGRAM_MIN_COUNT:-3}"
      export NGRAM_PACKED_CACHE="${NGRAM_PACKED_CACHE:-1}"
      export NGRAM_GLOBAL_CACHE="${NGRAM_GLOBAL_CACHE:-1}"
      ;;
    *)
      echo "unknown TTT candidate: $candidate" >&2
      return 1
      ;;
  esac
}

record_push_candidate_slug() {
  local arch_candidate="${1:-baseline}"
  local ttt_candidate="${2:-baseline}"
  if [[ -z "$arch_candidate" || "$arch_candidate" == "baseline" ]]; then
    if [[ -z "$ttt_candidate" || "$ttt_candidate" == "baseline" ]]; then
      printf '%s\n' "baseline"
    else
      printf '%s\n' "$ttt_candidate"
    fi
    return 0
  fi
  if [[ -z "$ttt_candidate" || "$ttt_candidate" == "baseline" ]]; then
    printf '%s\n' "$arch_candidate"
    return 0
  fi
  if [[ "$ttt_candidate" == bg3072_* && "$arch_candidate" == *bg3072* ]]; then
    printf '%s_%s\n' "$arch_candidate" "${ttt_candidate#bg3072_}"
    return 0
  fi
  printf '%s_%s\n' "$arch_candidate" "$ttt_candidate"
}

record_push_artifact_log_path() {
  local root_dir="$1"
  local candidate="$2"
  local log_dir="$root_dir/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs"
  case "$candidate" in
    baseline)
      printf '%s\n' "$log_dir/h200_artifact_ttt_baseline.txt"
      ;;
    tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_tttlr25.txt"
      ;;
    tttlr30)
      printf '%s\n' "$log_dir/h200_artifact_ttt_tttlr30.txt"
      ;;
    batch48)
      printf '%s\n' "$log_dir/h200_artifact_ttt_batch48.txt"
      ;;
    tttlr25_batch48)
      printf '%s\n' "$log_dir/h200_artifact_ttt_tttlr25_batch48.txt"
      ;;
    chunk16k)
      printf '%s\n' "$log_dir/h200_artifact_ttt_chunk16k.txt"
      ;;
    freeze2_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_freeze2_tttlr25.txt"
      ;;
    epochs2_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_epochs2_tttlr25.txt"
      ;;
    freeze2_epochs2_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_freeze2_epochs2_tttlr25.txt"
      ;;
    bg3072_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_bg3072_tttlr25.txt"
      ;;
    ngram659)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659.txt"
      ;;
    ngram659_conf06)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf06.txt"
      ;;
    ngram659_conf07)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf07.txt"
      ;;
    ngram659_warm_conf07)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_warm_conf07.txt"
      ;;
    ngram659_orderlam)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_orderlam.txt"
      ;;
    ngram659_warm_conf07_orderlam)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_warm_conf07_orderlam.txt"
      ;;
    lowrisk_ngram)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lowrisk.txt"
      ;;
    lam10_conf05_ngram)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lam10_conf05.txt"
      ;;
    ngram659_adapt)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt.txt"
      ;;
    ngram659_adapt_last2)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last2.txt"
      ;;
    ngram659_adapt_last4)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last4.txt"
      ;;
    lowrisk_ngram_adapt)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lowrisk_adapt.txt"
      ;;
    ngram659_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_tttlr25.txt"
      ;;
    ngram659_late2_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_late2_tttlr25.txt"
      ;;
    ngram659_late2_adamw5e4)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw5e4_late2.txt"
      ;;
    ngram659_late2_adamw1e4)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw1e4_late2.txt"
      ;;
    ngram659_adamw30ep_cosine)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw30ep_cosine.txt"
      ;;
    ngram659_adamw30ep_cosine_lr3e4)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lr3e4.txt"
      ;;
    lowrisk_ngram_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_lowrisk_tttlr25.txt"
      ;;
    *)
      echo "unknown artifact TTT candidate: $candidate" >&2
      return 1
      ;;
  esac
}

record_push_proxy_log_path() {
  local root_dir="$1"
  local arch_candidate="${2:-baseline}"
  local ttt_candidate="${3:-baseline}"
  local seed="${4:-1337}"
  local log_dir="$root_dir/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs"
  local slug
  slug="$(record_push_candidate_slug "$arch_candidate" "$ttt_candidate")"
  if [[ "$slug" == "baseline" ]]; then
    printf '%s\n' "$log_dir/h200_ttt_h100proxy7185_seed${seed}.txt"
    return 0
  fi
  printf '%s\n' "$log_dir/h200_ttt_h100proxy7185_${slug}_seed${seed}.txt"
}

record_push_ngram_log_path() {
  local root_dir="$1"
  local candidate="$2"
  local log_dir="$root_dir/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs"
  case "$candidate" in
    record659)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659.txt"
      ;;
    record659_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_smoke.txt"
      ;;
    record659_conf06)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf06.txt"
      ;;
    record659_conf06_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf06_smoke.txt"
      ;;
    record659_conf07)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf07.txt"
      ;;
    record659_conf07_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_conf07_smoke.txt"
      ;;
    lowrisk)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lowrisk.txt"
      ;;
    lowrisk_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lowrisk_smoke.txt"
      ;;
    lam10_conf05)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lam10_conf05.txt"
      ;;
    vr1_record659)
      printf '%s\n' "$log_dir/h200_artifact_ngram_vr1_record659.txt"
      ;;
    record659_adapt_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_smoke.txt"
      ;;
    record659_adapt)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt.txt"
      ;;
    record659_adapt_last2_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last2_smoke.txt"
      ;;
    record659_adapt_last2)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last2.txt"
      ;;
    record659_adapt_last4_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last4_smoke.txt"
      ;;
    record659_adapt_last4)
      printf '%s\n' "$log_dir/h200_artifact_ngram_record659_adapt_last4.txt"
      ;;
    lowrisk_adapt)
      printf '%s\n' "$log_dir/h200_artifact_ngram_lowrisk_adapt.txt"
      ;;
    *)
      echo "unknown artifact n-gram candidate: $candidate" >&2
      return 1
      ;;
  esac
}

record_push_ttt_ngram_log_path() {
  local root_dir="$1"
  local candidate="$2"
  local log_dir="$root_dir/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs"
  case "$candidate" in
    record659_tttlr25_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_tttlr25_smoke.txt"
      ;;
    record659_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_tttlr25.txt"
      ;;
    record659_late2_tttlr25_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_late2_tttlr25_smoke.txt"
      ;;
    record659_adamw5e4_late2_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw5e4_late2_smoke.txt"
      ;;
    record659_adamw1e4_late2_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw1e4_late2_smoke.txt"
      ;;
    record659_adamw5e4_late2)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_record659_adamw5e4_late2.txt"
      ;;
    lowrisk_tttlr25_smoke)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_lowrisk_tttlr25_smoke.txt"
      ;;
    lowrisk_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_lowrisk_tttlr25.txt"
      ;;
    vr1_record659_tttlr25)
      printf '%s\n' "$log_dir/h200_artifact_ttt_ngram_vr1_record659_tttlr25.txt"
      ;;
    *)
      echo "unknown artifact TTT+n-gram candidate: $candidate" >&2
      return 1
      ;;
  esac
}
