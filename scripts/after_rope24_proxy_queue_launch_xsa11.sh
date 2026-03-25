#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"

ROPE24_PROXY_CONF07_LOG="${ROPE24_PROXY_CONF07_LOG:-$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_rope24_seed1337.txt}"
XSA11_PROXY_TRAIN_LOG="${XSA11_PROXY_TRAIN_LOG:-$LOG_DIR/h200_ttt_h100proxy7185_xsa11_seed1337.txt}"
XSA11_PROXY_ARTIFACT_PT="${XSA11_PROXY_ARTIFACT_PT:-$RUN_DIR/final_model_h100proxy7185_xsa11_seed1337.pt}"
XSA11_PROXY_ARTIFACT_INT6="${XSA11_PROXY_ARTIFACT_INT6:-$RUN_DIR/final_model_h100proxy7185_xsa11_seed1337.int6.ptz}"
XSA11_RECORD674_LOG="${XSA11_RECORD674_LOG:-$LOG_DIR/h200_artifact_ngram_record674_h100proxy7185_xsa11_seed1337.txt}"
XSA11_RECORD674_LAM18_SMOKE_LOG="${XSA11_RECORD674_LAM18_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_lam18_h100proxy7185_xsa11_seed1337_smoke.txt}"
XSA11_RECORD674_LAM22_SMOKE_LOG="${XSA11_RECORD674_LAM22_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_lam22_h100proxy7185_xsa11_seed1337_smoke.txt}"
XSA11_RECORD674_MIN3_SMOKE_LOG="${XSA11_RECORD674_MIN3_SMOKE_LOG:-$LOG_DIR/h200_artifact_ngram_record674_min3_h100proxy7185_xsa11_seed1337_smoke.txt}"
XSA11_PROXY_CONF07_LOG="${XSA11_PROXY_CONF07_LOG:-$LOG_DIR/h200_artifact_ngram_record659_conf07_h100proxy7185_xsa11_seed1337.txt}"

cd "$ROOT_DIR"
source .venv/bin/activate

while ! rg -q '^final_ngram_eval_exact val_loss:' "$ROPE24_PROXY_CONF07_LOG" 2>/dev/null; do
  sleep 60
done

PROXY_TRAIN_LOG="$XSA11_PROXY_TRAIN_LOG" \
PROXY_ARTIFACT_PT="$XSA11_PROXY_ARTIFACT_PT" \
PROXY_ARTIFACT_INT6="$XSA11_PROXY_ARTIFACT_INT6" \
PROXY_RECORD674_LOG="$XSA11_RECORD674_LOG" \
PROXY_RECORD674_LAM18_SMOKE_LOG="$XSA11_RECORD674_LAM18_SMOKE_LOG" \
PROXY_RECORD674_LAM22_SMOKE_LOG="$XSA11_RECORD674_LAM22_SMOKE_LOG" \
PROXY_RECORD674_MIN3_SMOKE_LOG="$XSA11_RECORD674_MIN3_SMOKE_LOG" \
PROXY_CONF07_LOG="$XSA11_PROXY_CONF07_LOG" \
setsid bash "$ROOT_DIR/scripts/after_proxy_train_run_record674_then_conf07.sh" >/tmp/h200_after_xsa11_proxy_train_record674_then_conf07.log 2>&1 < /dev/null &

ARCH_CANDIDATE="xsa11" \
SEED="${SEED:-1337}" \
bash "$ROOT_DIR/scripts/icrn_h200_ttt_h100_proxy_candidate.sh" >"$XSA11_PROXY_TRAIN_LOG" 2>&1
