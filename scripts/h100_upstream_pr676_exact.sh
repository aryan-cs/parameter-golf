#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/../parameter-golf-pr676-worktree}"
PR676_RECORD_DIR="records/track_10min_16mb/2026-03-25_SwiGLU_LeakyReLU2_LegalTTT_ParallelMuon"
SEED="${SEED:-1337}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs}"
TIMED_MODE="${TIMED_MODE:-0}"

run_suffix=""
if [[ "$TIMED_MODE" == "1" ]]; then
  : "${WARMUP_STEPS:=0}"
  : "${VAL_LOSS_EVERY:=0}"
  : "${TRAIN_LOG_EVERY:=1000}"
  : "${MAX_WALLCLOCK_SECONDS:=596}"
  run_suffix="_timed"
fi

RUN_ID="${RUN_ID:-h100_upstream_pr676${run_suffix}_seed${SEED}}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${RUN_ID}.txt}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
  git -C "$ROOT_DIR" worktree add "$WORKTREE_DIR" pr676
fi
git config --global --add safe.directory "$WORKTREE_DIR" >/dev/null 2>&1 || true

cd "$ROOT_DIR"
source .venv/bin/activate

cd "$WORKTREE_DIR/$PR676_RECORD_DIR"

run_with_timer() {
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -f 'elapsed=%E' "$@"
  elif [[ -x /bin/time ]]; then
    /bin/time -f 'elapsed=%E' "$@"
  else
    TIMEFORMAT='elapsed=%R'
    time "$@"
  fi
}

run_with_timer env \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  WARMUP_STEPS="${WARMUP_STEPS:-20}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}" \
  NUM_LAYERS="${NUM_LAYERS:-11}" \
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
  XSA_LAST_N="${XSA_LAST_N:-4}" \
  EMA_ENABLED="${EMA_ENABLED:-1}" \
  EMA_DECAY="${EMA_DECAY:-0.997}" \
  SWA_ENABLED="${SWA_ENABLED:-1}" \
  SWA_EVERY="${SWA_EVERY:-50}" \
  ROPE_DIMS="${ROPE_DIMS:-16}" \
  LN_SCALE="${LN_SCALE:-1}" \
  LATE_QAT="${LATE_QAT:-1}" \
  LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}" \
  VE_ENABLED="${VE_ENABLED:-1}" \
  VE_DIM="${VE_DIM:-128}" \
  VE_LAYERS="${VE_LAYERS:-9,10}" \
  TTT_ENABLED="${TTT_ENABLED:-1}" \
  TTT_LR="${TTT_LR:-0.002}" \
  TTT_EPOCHS="${TTT_EPOCHS:-3}" \
  TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}" \
  TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}" \
  TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}" \
  TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}" \
  TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}" \
  MUON_WD="${MUON_WD:-0.04}" \
  ADAM_WD="${ADAM_WD:-0.04}" \
  MATRIX_LR="${MATRIX_LR:-0.025}" \
  SCALAR_LR="${SCALAR_LR:-0.025}" \
  TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}" \
  MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}" \
  MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}" \
  MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}" \
  WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}" \
  ITERATIONS="${ITERATIONS:-9000}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  USE_SWIGLU="${USE_SWIGLU:-1}" \
  SWIGLU_HALF_DIM="${SWIGLU_HALF_DIM:-1024}" \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee "$LOG_PATH"
