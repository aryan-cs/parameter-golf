#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
CANDIDATE="${CANDIDATE:-record659}"

cd "$ROOT_DIR"
source .venv/bin/activate

STRIDE="${STRIDE-}"
NGRAM_LAMBDA="${NGRAM_LAMBDA-}"
NGRAM_MAX_N="${NGRAM_MAX_N-}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD-}"
GATE_MODE="${GATE_MODE-}"
MIN_COUNT="${MIN_COUNT-}"
APPLY_MODE="${APPLY_MODE-}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE-}"
NGRAM_ADAPT_ENABLED="${NGRAM_ADAPT_ENABLED-}"
NGRAM_ADAPT_LR="${NGRAM_ADAPT_LR-}"
NGRAM_ADAPT_DECAY="${NGRAM_ADAPT_DECAY-}"
NGRAM_ADAPT_LAST_N_BLOCKS="${NGRAM_ADAPT_LAST_N_BLOCKS-}"
CONFIDENCE_SCHEDULE="${CONFIDENCE_SCHEDULE-}"
ORDER_LAMBDAS="${ORDER_LAMBDAS-}"
PACKED_CACHE="${PACKED_CACHE-}"
BATCH_SEQS="${BATCH_SEQS-}"
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE-}"
VALUE_RESIDUAL="${VALUE_RESIDUAL-}"
MAX_WINDOWS="${MAX_WINDOWS-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
ARTIFACT_PATH="${ARTIFACT_PATH:-}"
TEMPLATE_PATH="${TEMPLATE_PATH:-}"
TRAIN_GPT_PATH="${TRAIN_GPT_PATH:-}"
EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_KIND="${CACHE_KIND-}"
HASHED_BUCKETS="${HASHED_BUCKETS-}"
NGRAM_MIN_ORDER="${NGRAM_MIN_ORDER-}"
NGRAM_ADAPTIVE_ALPHA="${NGRAM_ADAPTIVE_ALPHA-}"
NGRAM_ALPHA_MIN="${NGRAM_ALPHA_MIN-}"
NGRAM_ALPHA_MAX="${NGRAM_ALPHA_MAX-}"
NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER-}"
NGRAM_ENTROPY_SCALE="${NGRAM_ENTROPY_SCALE-}"
NGRAM_MAX_SECONDS="${NGRAM_MAX_SECONDS-}"
HEDGE_ENABLED="${HEDGE_ENABLED-}"
HEDGE_ETA="${HEDGE_ETA-}"
HEDGE_NEURAL_BIAS="${HEDGE_NEURAL_BIAS-}"
MIXER_ETA="${MIXER_ETA-}"
MIXER_NEURAL_BIAS="${MIXER_NEURAL_BIAS-}"
MIXER_TRIGRAM_BUCKETS="${MIXER_TRIGRAM_BUCKETS-}"
MIXER_WARMUP_TOKENS="${MIXER_WARMUP_TOKENS-}"

case "$CANDIDATE" in
  record659)
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659.txt}"
    ;;
  record659_smoke)
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_smoke.txt}"
    ;;
  record659_lamcool_smoke)
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lamcool_smoke.txt}"
    ;;
  record659_lamcool)
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lamcool.txt}"
    ;;
  record659_conf06)
    CONFIDENCE_THRESHOLD="0.6"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf06.txt}"
    ;;
  record659_conf06_smoke)
    CONFIDENCE_THRESHOLD="0.6"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf06_smoke.txt}"
    ;;
  record659_conf07)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07.txt}"
    ;;
  record659_conf07_hedge)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    HEDGE_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge.txt}"
    ;;
  record659_conf07_smoke)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_smoke.txt}"
    ;;
  record659_conf07_hedge_smoke)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    HEDGE_ENABLED="1"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge_smoke.txt}"
    ;;
  record659_conf07_hedge_eta05_smoke)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    HEDGE_ENABLED="1"
    HEDGE_ETA="0.05"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge_eta05_smoke.txt}"
    ;;
  record659_conf07_hedge_eta20_smoke)
    STRIDE="128"
    NGRAM_LAMBDA="0.15"
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="3"
    APPLY_MODE="improve_only"
    CACHE_KIND="exact"
    HEDGE_ENABLED="1"
    HEDGE_ETA="0.20"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge_eta20_smoke.txt}"
    ;;
  record659_latecool_conf07_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07_smoke.txt}"
    ;;
  record659_latecool_conf07)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07.txt}"
    ;;
  record659_latecool_conf07_lamtail_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    LAMBDA_SCHEDULE="0.00:0.15,0.72:0.12,0.80:0.09,0.90:0.06"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07_lamtail_smoke.txt}"
    ;;
  record659_latecool_conf07_lamtail)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    LAMBDA_SCHEDULE="0.00:0.15,0.72:0.12,0.80:0.09,0.90:0.06"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07_lamtail.txt}"
    ;;
  record659_latecool_conf07_min4_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07_min4_smoke.txt}"
    ;;
  record659_latecool_conf07_min4)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.72:0.65,0.80:0.60,0.90:0.55"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_latecool_conf07_min4.txt}"
    ;;
  record659_conf07_lamcool_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_lamcool_smoke.txt}"
    ;;
  record659_conf07_lamcool)
    CONFIDENCE_THRESHOLD="0.7"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_lamcool.txt}"
    ;;
  record659_cool_conf07)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07.txt}"
    ;;
  record659_cool_conf07_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07_smoke.txt}"
    ;;
  record659_cool_conf07_min4)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07_min4.txt}"
    ;;
  record659_cool_conf07_min4_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07_min4_smoke.txt}"
    ;;
  record659_cool_conf07_lamcool_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07_lamcool_smoke.txt}"
    ;;
  record659_cool_conf07_lamcool)
    CONFIDENCE_THRESHOLD="0.7"
    CONFIDENCE_SCHEDULE="0.00:0.70,0.50:0.65,0.65:0.60,0.80:0.55"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_cool_conf07_lamcool.txt}"
    ;;
  record659_conf08)
    CONFIDENCE_THRESHOLD="0.8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf08.txt}"
    ;;
  record659_conf08_smoke)
    CONFIDENCE_THRESHOLD="0.8"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf08_smoke.txt}"
    ;;
  record659_conf07_min4_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_min4_smoke.txt}"
    ;;
  record659_conf07_min4)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_min4.txt}"
    ;;
  record659_conf07_min5_smoke)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="5"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_min5_smoke.txt}"
    ;;
  record659_conf07_min5)
    CONFIDENCE_THRESHOLD="0.7"
    MIN_COUNT="5"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_conf07_min5.txt}"
    ;;
  record659_tgate30_smoke)
    GATE_MODE="target"
    CONFIDENCE_THRESHOLD="0.3"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_tgate30_smoke.txt}"
    ;;
  record659_tgate40_smoke)
    GATE_MODE="target"
    CONFIDENCE_THRESHOLD="0.4"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_tgate40_smoke.txt}"
    ;;
  record659_tgate40_min4_smoke)
    GATE_MODE="target"
    CONFIDENCE_THRESHOLD="0.4"
    MIN_COUNT="4"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_tgate40_min4_smoke.txt}"
    ;;
  record659_tgate40_min4)
    GATE_MODE="target"
    CONFIDENCE_THRESHOLD="0.4"
    MIN_COUNT="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_tgate40_min4.txt}"
    ;;
  record659_lam20_conf07_smoke)
    NGRAM_LAMBDA="0.20"
    CONFIDENCE_THRESHOLD="0.7"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lam20_conf07_smoke.txt}"
    ;;
  record659_lam20_conf07)
    NGRAM_LAMBDA="0.20"
    CONFIDENCE_THRESHOLD="0.7"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lam20_conf07.txt}"
    ;;
  record659_lam20_conf08_smoke)
    NGRAM_LAMBDA="0.20"
    CONFIDENCE_THRESHOLD="0.8"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lam20_conf08_smoke.txt}"
    ;;
  record659_lam20_conf08)
    NGRAM_LAMBDA="0.20"
    CONFIDENCE_THRESHOLD="0.8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_lam20_conf08.txt}"
    ;;
  record674_smoke)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674_smoke.txt}"
    ;;
  record674)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674.txt}"
    ;;
  record674_hedge)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    HEDGE_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674_hedge.txt}"
    ;;
  record674_proxy7185)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_seed1337.txt}"
    ;;
  record674_hedge_smoke)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    HEDGE_ENABLED="1"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674_hedge_smoke.txt}"
    ;;
  record674_hedge_proxy7185)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.20}"
    CONFIDENCE_THRESHOLD="1.0"
    MIN_COUNT="${MIN_COUNT:-2}"
    APPLY_MODE="always"
    CACHE_KIND="hashed"
    HEDGE_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record674_hedge_h100proxy7185_seed1337.txt}"
    ;;
  record753_smoke)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.30}"
    NGRAM_MAX_N="${NGRAM_MAX_N:-7}"
    NGRAM_MIN_ORDER="${NGRAM_MIN_ORDER:-2}"
    MIN_COUNT="${MIN_COUNT:-2}"
    CACHE_KIND="hashed_backoff"
    APPLY_MODE="${APPLY_MODE:-always}"
    NGRAM_ADAPTIVE_ALPHA="${NGRAM_ADAPTIVE_ALPHA:-1}"
    NGRAM_ALPHA_MIN="${NGRAM_ALPHA_MIN:-0.05}"
    NGRAM_ALPHA_MAX="${NGRAM_ALPHA_MAX:-0.60}"
    NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER:-4.0}"
    NGRAM_ENTROPY_SCALE="${NGRAM_ENTROPY_SCALE:-2.0}"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record753_smoke.txt}"
    ;;
  record753)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.30}"
    NGRAM_MAX_N="${NGRAM_MAX_N:-7}"
    NGRAM_MIN_ORDER="${NGRAM_MIN_ORDER:-2}"
    MIN_COUNT="${MIN_COUNT:-2}"
    CACHE_KIND="hashed_backoff"
    APPLY_MODE="${APPLY_MODE:-always}"
    NGRAM_ADAPTIVE_ALPHA="${NGRAM_ADAPTIVE_ALPHA:-1}"
    NGRAM_ALPHA_MIN="${NGRAM_ALPHA_MIN:-0.05}"
    NGRAM_ALPHA_MAX="${NGRAM_ALPHA_MAX:-0.60}"
    NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER:-4.0}"
    NGRAM_ENTROPY_SCALE="${NGRAM_ENTROPY_SCALE:-2.0}"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record753.txt}"
    ;;
  record753_proxy7185)
    STRIDE="64"
    NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.30}"
    NGRAM_MAX_N="${NGRAM_MAX_N:-7}"
    NGRAM_MIN_ORDER="${NGRAM_MIN_ORDER:-2}"
    MIN_COUNT="${MIN_COUNT:-2}"
    CACHE_KIND="hashed_backoff"
    APPLY_MODE="${APPLY_MODE:-always}"
    NGRAM_ADAPTIVE_ALPHA="${NGRAM_ADAPTIVE_ALPHA:-1}"
    NGRAM_ALPHA_MIN="${NGRAM_ALPHA_MIN:-0.05}"
    NGRAM_ALPHA_MAX="${NGRAM_ALPHA_MAX:-0.60}"
    NGRAM_ENTROPY_CENTER="${NGRAM_ENTROPY_CENTER:-4.0}"
    NGRAM_ENTROPY_SCALE="${NGRAM_ENTROPY_SCALE:-2.0}"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record753_h100proxy7185_seed1337.txt}"
    ;;
  record688_mixer5_smoke)
    STRIDE="32"
    CACHE_KIND="mixer5"
    MIXER_ETA="0.10"
    MIXER_NEURAL_BIAS="2.0"
    MIXER_TRIGRAM_BUCKETS="65536"
    MIXER_WARMUP_TOKENS="10000"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record688_mixer5_smoke.txt}"
    ;;
  record688_mixer5)
    STRIDE="32"
    CACHE_KIND="mixer5"
    MIXER_ETA="0.10"
    MIXER_NEURAL_BIAS="2.0"
    MIXER_TRIGRAM_BUCKETS="65536"
    MIXER_WARMUP_TOKENS="10000"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record688_mixer5.txt}"
    ;;
  record688_mixer5_eta05_smoke)
    STRIDE="32"
    CACHE_KIND="mixer5"
    MIXER_ETA="0.05"
    MIXER_NEURAL_BIAS="2.0"
    MIXER_TRIGRAM_BUCKETS="65536"
    MIXER_WARMUP_TOKENS="10000"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record688_mixer5_eta05_smoke.txt}"
    ;;
  record688_mixer5_eta20_smoke)
    STRIDE="32"
    CACHE_KIND="mixer5"
    MIXER_ETA="0.20"
    MIXER_NEURAL_BIAS="2.0"
    MIXER_TRIGRAM_BUCKETS="65536"
    MIXER_WARMUP_TOKENS="10000"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record688_mixer5_eta20_smoke.txt}"
    ;;
  record659_warm_conf07)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07.txt}"
    ;;
  record659_warm_conf07_smoke)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_smoke.txt}"
    ;;
  record659_orderlam)
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_orderlam.txt}"
    ;;
  record659_orderlam_smoke)
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_orderlam_smoke.txt}"
    ;;
  record659_warm_conf07_orderlam)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_orderlam.txt}"
    ;;
  record659_warm_conf07_orderlam_smoke)
    CONFIDENCE_SCHEDULE="0.00:0.50,0.20:0.60,0.40:0.70"
    ORDER_LAMBDAS="2:0.08,3:0.12,4:0.17,5:0.22"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_warm_conf07_orderlam_smoke.txt}"
    ;;
  lowrisk)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk.txt}"
    ;;
  lowrisk_smoke)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    MAX_WINDOWS="128"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk_smoke.txt}"
    ;;
  lam10_conf05)
    NGRAM_LAMBDA="0.10"
    CONFIDENCE_THRESHOLD="0.5"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lam10_conf05.txt}"
    ;;
  vr1_record659)
    VALUE_RESIDUAL="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_vr1_record659.txt}"
    ;;
  record659_adapt_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_smoke.txt}"
    ;;
  record659_adapt)
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt.txt}"
    ;;
  record659_adapt_last2_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="2"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last2_smoke.txt}"
    ;;
  record659_adapt_last2)
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="2"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last2.txt}"
    ;;
  record659_adapt_last4_smoke)
    MAX_WINDOWS="128"
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last4_smoke.txt}"
    ;;
  record659_adapt_last4)
    NGRAM_ADAPT_ENABLED="1"
    NGRAM_ADAPT_LAST_N_BLOCKS="4"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_record659_adapt_last4.txt}"
    ;;
  lowrisk_adapt)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    NGRAM_ADAPT_ENABLED="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ngram_lowrisk_adapt.txt}"
    ;;
  *)
    echo "unknown artifact ngram candidate: $CANDIDATE" >&2
    exit 1
    ;;
esac

: "${STRIDE:=128}"
: "${NGRAM_LAMBDA:=0.15}"
: "${NGRAM_MAX_N:=5}"
: "${CONFIDENCE_THRESHOLD:=0.5}"
: "${GATE_MODE:=max}"
: "${MIN_COUNT:=3}"
: "${APPLY_MODE:=improve_only}"
: "${LAMBDA_SCHEDULE:=}"
: "${NGRAM_ADAPT_ENABLED:=0}"
: "${NGRAM_ADAPT_LR:=0.0003}"
: "${NGRAM_ADAPT_DECAY:=0.001}"
: "${NGRAM_ADAPT_LAST_N_BLOCKS:=3}"
: "${CONFIDENCE_SCHEDULE:=}"
: "${ORDER_LAMBDAS:=}"
: "${PACKED_CACHE:=1}"
: "${BATCH_SEQS:=32}"
: "${BIGRAM_VOCAB_SIZE:=1536}"
: "${VALUE_RESIDUAL:=0}"
: "${MAX_WINDOWS:=0}"
: "${CACHE_KIND:=exact}"
: "${HASHED_BUCKETS:=4194304}"
: "${NGRAM_MIN_ORDER:=2}"
: "${NGRAM_ADAPTIVE_ALPHA:=0}"
: "${NGRAM_ALPHA_MIN:=0.05}"
: "${NGRAM_ALPHA_MAX:=0.60}"
: "${NGRAM_ENTROPY_CENTER:=4.0}"
: "${NGRAM_ENTROPY_SCALE:=2.0}"
: "${NGRAM_MAX_SECONDS:=0.0}"
: "${HEDGE_ENABLED:=0}"
: "${HEDGE_ETA:=0.10}"
: "${HEDGE_NEURAL_BIAS:=2.0}"
: "${MIXER_ETA:=0.10}"
: "${MIXER_NEURAL_BIAS:=2.0}"
: "${MIXER_TRIGRAM_BUCKETS:=65536}"
: "${MIXER_WARMUP_TOKENS:=10000}"

if [[ "$SKIP_COMPLETED" == "1" && -f "$LOG_PATH" ]] && rg -q "final_ngram_eval_exact|final_int6_sliding_window_ngram[0-9]+_exact" "$LOG_PATH"; then
  echo "skipping completed ngram candidate '$CANDIDATE' at $LOG_PATH"
  exit 0
fi

rm -f "$LOG_PATH"

args=(
  scripts/eval_ngram_cache_artifact.py
  --run-dir "$RUN_DIR"
  --log-path "$LOG_PATH"
  --batch-seqs "$BATCH_SEQS"
  --bigram-vocab-size "$BIGRAM_VOCAB_SIZE"
  --value-residual "$VALUE_RESIDUAL"
  --stride "$STRIDE"
  --ngram-lambda "$NGRAM_LAMBDA"
  --ngram-max-n "$NGRAM_MAX_N"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --gate-mode "$GATE_MODE"
  --min-count "$MIN_COUNT"
  --apply-mode "$APPLY_MODE"
  --cache-kind "$CACHE_KIND"
  --hashed-buckets "$HASHED_BUCKETS"
  --ngram-min-order "$NGRAM_MIN_ORDER"
  --ngram-alpha-min "$NGRAM_ALPHA_MIN"
  --ngram-alpha-max "$NGRAM_ALPHA_MAX"
  --ngram-entropy-center "$NGRAM_ENTROPY_CENTER"
  --ngram-entropy-scale "$NGRAM_ENTROPY_SCALE"
  --ngram-max-seconds "$NGRAM_MAX_SECONDS"
  --hedge-eta "$HEDGE_ETA"
  --hedge-neural-bias "$HEDGE_NEURAL_BIAS"
  --mixer-eta "$MIXER_ETA"
  --mixer-neural-bias "$MIXER_NEURAL_BIAS"
  --mixer-trigram-buckets "$MIXER_TRIGRAM_BUCKETS"
  --mixer-warmup-tokens "$MIXER_WARMUP_TOKENS"
  --lambda-schedule "$LAMBDA_SCHEDULE"
  --confidence-schedule "$CONFIDENCE_SCHEDULE"
  --order-lambdas "$ORDER_LAMBDAS"
  --ngram-adapt-lr "$NGRAM_ADAPT_LR"
  --ngram-adapt-decay "$NGRAM_ADAPT_DECAY"
  --ngram-adapt-last-n-blocks "$NGRAM_ADAPT_LAST_N_BLOCKS"
  --max-windows "$MAX_WINDOWS"
)

if [[ "$NGRAM_ADAPT_ENABLED" == "1" ]]; then
  args+=(--ngram-adapt-enabled)
fi
if [[ "$HEDGE_ENABLED" == "1" ]]; then
  args+=(--hedge-enabled)
fi
if [[ "$PACKED_CACHE" == "1" ]]; then
  args+=(--packed-cache)
fi
if [[ "$NGRAM_ADAPTIVE_ALPHA" == "1" ]]; then
  args+=(--ngram-adaptive-alpha)
fi
if [[ -n "$ARTIFACT_PATH" ]]; then
  args+=(--artifact-path "$ARTIFACT_PATH")
fi
if [[ -n "$TEMPLATE_PATH" ]]; then
  args+=(--template-path "$TEMPLATE_PATH")
fi
if [[ -n "$TRAIN_GPT_PATH" ]]; then
  args+=(--train-gpt-path "$TRAIN_GPT_PATH")
fi
if [[ -n "$EVAL_TOKENIZER_PATH" ]]; then
  args+=(--tokenizer-path "$EVAL_TOKENIZER_PATH")
fi
if [[ -n "$EVAL_DATA_PATH" ]]; then
  args+=(--data-path "$EVAL_DATA_PATH")
fi

exec python "${args[@]}"
