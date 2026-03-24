#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
cd "$RUN_DIR"

export DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export RUN_ID="${RUN_ID:-h100_repro_leaky_ttt_parallel_muon_seed${SEED}}"

export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-786432}"
export ITERATIONS="${ITERATIONS:-9000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export EXTRA_STRIDE64_FINAL_EVAL="${EXTRA_STRIDE64_FINAL_EVAL:-0}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"

export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"

export TTT_ENABLED="${TTT_ENABLED:-1}"
export TTT_LR="${TTT_LR:-0.002}"
export TTT_EPOCHS="${TTT_EPOCHS:-3}"
export TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"
export TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
export TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
export TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"

exec torchrun --standalone --nproc_per_node=8 train_gpt.py
