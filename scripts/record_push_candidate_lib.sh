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
