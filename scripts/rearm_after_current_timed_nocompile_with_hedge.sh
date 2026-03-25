#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
SEED="${SEED:-1337}"
UPSTREAM_WAIT_PATTERN="${UPSTREAM_WAIT_PATTERN:-^final_int6_sliding_window_ngram5_exact val_loss:}"
NGRAM_WAIT_PATTERN="${NGRAM_WAIT_PATTERN:-^final_ngram_eval_exact val_loss:}"

cd "$ROOT_DIR"
pkill -f '/scripts/after_log_launch_script.sh' || true

CURRENT_LOG="$LOG_DIR/h200_upstream_pr674_proxy7185_timed_nocompile_seed${SEED}.txt"
HEDGE_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge_smoke.txt"
HEDGE_FULL_LOG="$LOG_DIR/h200_artifact_ngram_record659_conf07_hedge.txt"
MIXER5_SMOKE_LOG="$LOG_DIR/h200_artifact_ngram_record688_mixer5_smoke.txt"
MIXER5_FULL_LOG="$LOG_DIR/h200_artifact_ngram_record688_mixer5.txt"
UPSTREAM_PR674_MIXER5_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_mixer5_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_ENHATTN_MIXER5_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_enhattn_mixer5_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_ENHATTN_CROWNQ_MIXER5_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_enhattn_crownq_mixer5_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_hedgemix_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_CROWNQ_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_crownq_proxy7185_timed_nocompile_seed${SEED}.txt"
UPSTREAM_PR674_CROWNQ_MIXER5_TIMED_NOCOMPILE_LOG="$LOG_DIR/h200_upstream_pr674_crownq_mixer5_proxy7185_timed_nocompile_seed${SEED}.txt"

WAIT_LOG="$CURRENT_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="record659_conf07_hedge_smoke" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh" \
TARGET_LOG_PATH="$HEDGE_SMOKE_LOG" \
TARGET_ENV_ASSIGNMENTS="CANDIDATE=record659_conf07_hedge_smoke" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$HEDGE_SMOKE_LOG" \
NEXT_WAIT_PATTERN="$NGRAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="record659_conf07_hedge" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh" \
NEXT_LOG_PATH="$HEDGE_FULL_LOG" \
NEXT_TARGET_ENV_ASSIGNMENTS="CANDIDATE=record659_conf07_hedge" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_pr674_timed_nocompile_launch_record659_conf07_hedge_smoke.log 2>&1 < /dev/null &

WAIT_LOG="$HEDGE_FULL_LOG" \
WAIT_PATTERN="$NGRAM_WAIT_PATTERN" \
TARGET_LABEL="record688_mixer5_smoke" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh" \
TARGET_LOG_PATH="$MIXER5_SMOKE_LOG" \
TARGET_ENV_ASSIGNMENTS="CANDIDATE=record688_mixer5_smoke" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$MIXER5_SMOKE_LOG" \
NEXT_WAIT_PATTERN="$NGRAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="record688_mixer5" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_artifact_ngram_candidate.sh" \
NEXT_LOG_PATH="$MIXER5_FULL_LOG" \
NEXT_TARGET_ENV_ASSIGNMENTS="CANDIDATE=record688_mixer5" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_record659_conf07_hedge_launch_record688_mixer5_smoke.log 2>&1 < /dev/null &

WAIT_LOG="$MIXER5_FULL_LOG" \
WAIT_PATTERN="$NGRAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_mixer5_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_mixer5_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_MIXER5_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_mixer5_proxy7185_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR674_MIXER5_TIMED_NOCOMPILE_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr674_enhattn_mixer5_timed_nocompile_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_enhattn_mixer5_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR674_ENHATTN_MIXER5_TIMED_NOCOMPILE_LOG" \
NEXT_TARGET_RUN_ID="h200_upstream_pr674_enhattn_mixer5_proxy7185_timed_nocompile_seed${SEED}" \
NEXT_TARGET_SEED="$SEED" \
NEXT_TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_record688_mixer5_launch_upstream_pr674_mixer5_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_ENHATTN_MIXER5_TIMED_NOCOMPILE_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_enhattn_crownq_mixer5_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_enhattn_crownq_mixer5_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_ENHATTN_CROWNQ_MIXER5_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_enhattn_crownq_mixer5_proxy7185_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_enhattn_mixer5_launch_upstream_pr674_enhattn_crownq_mixer5_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_ENHATTN_CROWNQ_MIXER5_TIMED_NOCOMPILE_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_hedgemix_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_hedgemix_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_hedgemix_proxy7185_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
NEXT_WAIT_LOG="$UPSTREAM_PR674_HEDGEMIX_TIMED_NOCOMPILE_LOG" \
NEXT_WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
NEXT_TARGET_LABEL="upstream_pr674_crownq_timed_nocompile_exact" \
NEXT_TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_crownq_proxy.sh" \
NEXT_LOG_PATH="$UPSTREAM_PR674_CROWNQ_TIMED_NOCOMPILE_LOG" \
NEXT_TARGET_RUN_ID="h200_upstream_pr674_crownq_proxy7185_timed_nocompile_seed${SEED}" \
NEXT_TARGET_SEED="$SEED" \
NEXT_TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
NEXT_TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_enhattn_crownq_mixer5_launch_upstream_pr674_hedgemix_timed_nocompile.log 2>&1 < /dev/null &

WAIT_LOG="$UPSTREAM_PR674_CROWNQ_TIMED_NOCOMPILE_LOG" \
WAIT_PATTERN="$UPSTREAM_WAIT_PATTERN" \
TARGET_LABEL="upstream_pr674_crownq_mixer5_timed_nocompile_exact" \
TARGET_SCRIPT="$ROOT_DIR/scripts/icrn_h200_upstream_pr674_crownq_mixer5_proxy.sh" \
TARGET_LOG_PATH="$UPSTREAM_PR674_CROWNQ_MIXER5_TIMED_NOCOMPILE_LOG" \
TARGET_RUN_ID="h200_upstream_pr674_crownq_mixer5_proxy7185_timed_nocompile_seed${SEED}" \
TARGET_SEED="$SEED" \
TARGET_ENV_ASSIGNMENTS="TIMED_MODE=1 COMPILE_ENABLED=0" \
TARGET_SKIP_IF_LOG_EXISTS="1" \
setsid bash "$ROOT_DIR/scripts/after_log_launch_script.sh" >/tmp/h200_after_upstream_pr674_crownq_launch_upstream_pr674_crownq_mixer5_timed_nocompile.log 2>&1 < /dev/null &
