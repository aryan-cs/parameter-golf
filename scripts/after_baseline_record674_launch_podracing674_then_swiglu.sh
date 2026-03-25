#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

WAIT_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_seed1337.txt" \
TARGET_ARCH_CANDIDATE="podracing674" \
TARGET_PROXY_TRAIN_LOG="$LOG_DIR/h200_ttt_h100proxy7185_podracing674_seed1337.txt" \
TARGET_PROXY_ARTIFACT_PT="$RUN_DIR/final_model_h100proxy7185_podracing674_seed1337.pt" \
TARGET_PROXY_ARTIFACT_INT6="$RUN_DIR/final_model_h100proxy7185_podracing674_seed1337.int6.ptz" \
TARGET_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_podracing674_seed1337.txt" \
TARGET_RECORD674_LAM18_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185_podracing674_seed1337_smoke.txt" \
TARGET_RECORD674_LAM22_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185_podracing674_seed1337_smoke.txt" \
TARGET_RECORD674_MIN3_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185_podracing674_seed1337_smoke.txt" \
TARGET_CONF07_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_podracing674_seed1337.txt" \
NEXT_ARCH_CANDIDATE="swiglu" \
NEXT_WAIT_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_podracing674_seed1337.txt" \
NEXT_PROXY_TRAIN_LOG="$LOG_DIR/h200_ttt_h100proxy7185_swiglu_seed1337.txt" \
NEXT_PROXY_ARTIFACT_PT="$RUN_DIR/final_model_h100proxy7185_swiglu_seed1337.pt" \
NEXT_PROXY_ARTIFACT_INT6="$RUN_DIR/final_model_h100proxy7185_swiglu_seed1337.int6.ptz" \
NEXT_RECORD674_LOG="$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_swiglu_seed1337.txt" \
NEXT_RECORD674_LAM18_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185_swiglu_seed1337_smoke.txt" \
NEXT_RECORD674_LAM22_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185_swiglu_seed1337_smoke.txt" \
NEXT_RECORD674_MIN3_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185_swiglu_seed1337_smoke.txt" \
NEXT_CONF07_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_swiglu_seed1337.txt" \
RUN_NEARBY_SMOKES="${RUN_NEARBY_SMOKES:-1}" \
RUN_CONF07_TAIL="${RUN_CONF07_TAIL:-0}" \
bash "$ROOT_DIR/scripts/after_record674_launch_arch.sh"
