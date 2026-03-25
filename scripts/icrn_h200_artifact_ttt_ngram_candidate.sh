#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback"
LOG_DIR="$RUN_DIR/logs"
CANDIDATE="${CANDIDATE:-record659_tttlr25_smoke}"

cd "$ROOT_DIR"
source .venv/bin/activate

BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}"
VALUE_RESIDUAL="${VALUE_RESIDUAL:-0}"
TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"
TTT_LAST_N_BLOCKS="${TTT_LAST_N_BLOCKS:-0}"
TTT_LR="${TTT_LR:-0.0025}"
TTT_EPOCHS="${TTT_EPOCHS:-3}"
TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-32768}"
TTT_OPTIMIZER="${TTT_OPTIMIZER:-sgd}"
TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-0.0}"
TTT_BETA1="${TTT_BETA1:-0.9}"
TTT_BETA2="${TTT_BETA2:-0.999}"
TTT_MOMENTUM="${TTT_MOMENTUM:-0.9}"
TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-1.0}"
TTT_SCHEDULE="${TTT_SCHEDULE:-chunk_cosine}"
TTT_LR_GROUPING="${TTT_LR_GROUPING:-role}"
TTT_PROJ_LR_MULT="${TTT_PROJ_LR_MULT:-3.0}"
TTT_FC_LR_MULT="${TTT_FC_LR_MULT:-0.5}"
TTT_OTHER_LR_MULT="${TTT_OTHER_LR_MULT:-1.0}"
BATCH_SEQS="${BATCH_SEQS:-32}"
STRIDE="${STRIDE:-64}"
NGRAM_LAMBDA="${NGRAM_LAMBDA:-0.15}"
NGRAM_MAX_N="${NGRAM_MAX_N:-5}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
MIN_COUNT="${MIN_COUNT:-3}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-}"
CONFIDENCE_SCHEDULE="${CONFIDENCE_SCHEDULE:-}"
ORDER_LAMBDAS="${ORDER_LAMBDAS:-}"
PACKED_CACHE="${PACKED_CACHE:-1}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "$CANDIDATE" in
  record659_tttlr25_smoke)
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_tttlr25_smoke.txt}"
    ;;
  record659_tttlr25)
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_tttlr25.txt}"
    ;;
  record659_late2_tttlr25_smoke)
    TTT_LAST_N_BLOCKS="2"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_late2_tttlr25_smoke.txt}"
    ;;
  record659_late2_tttlr25)
    TTT_LAST_N_BLOCKS="2"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_late2_tttlr25.txt}"
    ;;
  record659_adamw5e4_late2_smoke)
    TTT_LAST_N_BLOCKS="2"
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw5e4_late2_smoke.txt}"
    ;;
  record659_adamw1e4_late2_smoke)
    TTT_LAST_N_BLOCKS="2"
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0001"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw1e4_late2_smoke.txt}"
    ;;
  record659_adamw1e4_late2)
    TTT_LAST_N_BLOCKS="2"
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0001"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw1e4_late2.txt}"
    ;;
  record659_adamw5e4_late2)
    TTT_LAST_N_BLOCKS="2"
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw5e4_late2.txt}"
    ;;
  record659_adamw30ep_cosine_smoke)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_smoke.txt}"
    ;;
  record659_adamw30ep_cosine_lr3e4_smoke)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0003"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lr3e4_smoke.txt}"
    ;;
  record659_adamw12ep_cosine_smoke)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    TTT_EPOCHS="12"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw12ep_cosine_smoke.txt}"
    ;;
  record659_adamw30ep_cosine)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine.txt}"
    ;;
  record659_adamw30ep_cosine_lamcool_smoke)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lamcool_smoke.txt}"
    ;;
  record659_adamw30ep_cosine_lamcool)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0005"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    LAMBDA_SCHEDULE="0.00:0.15,0.50:0.12,0.65:0.09,0.80:0.06"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lamcool.txt}"
    ;;
  record659_adamw30ep_cosine_lr3e4)
    TTT_OPTIMIZER="adamw"
    TTT_LR="0.0003"
    TTT_EPOCHS="30"
    TTT_SCHEDULE="step_cosine"
    TTT_LR_GROUPING="pr672"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_record659_adamw30ep_cosine_lr3e4.txt}"
    ;;
  lowrisk_tttlr25_smoke)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    MAX_CHUNKS="8"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_lowrisk_tttlr25_smoke.txt}"
    ;;
  lowrisk_tttlr25)
    NGRAM_LAMBDA="0.05"
    CONFIDENCE_THRESHOLD="0.7"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_lowrisk_tttlr25.txt}"
    ;;
  vr1_record659_tttlr25)
    VALUE_RESIDUAL="1"
    LOG_PATH="${LOG_PATH:-$LOG_DIR/h200_artifact_ttt_ngram_vr1_record659_tttlr25.txt}"
    ;;
  *)
    echo "unknown artifact TTT+ngram candidate: $CANDIDATE" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_COMPLETED" == "1" && -f "$LOG_PATH" ]] && rg -q "legal_ttt_ngram_exact" "$LOG_PATH"; then
  echo "skipping completed TTT+ngram candidate '$CANDIDATE' at $LOG_PATH"
  exit 0
fi

rm -f "$LOG_PATH"

exec python scripts/eval_ngram_ttt_artifact.py \
  --run-dir "$RUN_DIR" \
  --log-path "$LOG_PATH" \
  --bigram-vocab-size "$BIGRAM_VOCAB_SIZE" \
  --value-residual "$VALUE_RESIDUAL" \
  --ttt-freeze-blocks "$TTT_FREEZE_BLOCKS" \
  --ttt-last-n-blocks "$TTT_LAST_N_BLOCKS" \
  --ttt-lr "$TTT_LR" \
  --ttt-epochs "$TTT_EPOCHS" \
  --ttt-chunk-tokens "$TTT_CHUNK_TOKENS" \
  --ttt-optimizer "$TTT_OPTIMIZER" \
  --ttt-weight-decay "$TTT_WEIGHT_DECAY" \
  --ttt-beta1 "$TTT_BETA1" \
  --ttt-beta2 "$TTT_BETA2" \
  --ttt-momentum "$TTT_MOMENTUM" \
  --ttt-grad-clip "$TTT_GRAD_CLIP" \
  --ttt-schedule "$TTT_SCHEDULE" \
  --ttt-lr-grouping "$TTT_LR_GROUPING" \
  --ttt-proj-lr-mult "$TTT_PROJ_LR_MULT" \
  --ttt-fc-lr-mult "$TTT_FC_LR_MULT" \
  --ttt-other-lr-mult "$TTT_OTHER_LR_MULT" \
  --batch-seqs "$BATCH_SEQS" \
  --stride "$STRIDE" \
  --ngram-lambda "$NGRAM_LAMBDA" \
  --ngram-max-n "$NGRAM_MAX_N" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD" \
  --min-count "$MIN_COUNT" \
  --lambda-schedule "$LAMBDA_SCHEDULE" \
  --confidence-schedule "$CONFIDENCE_SCHEDULE" \
  --order-lambdas "$ORDER_LAMBDAS" \
  $( [[ "$PACKED_CACHE" == "1" ]] && printf '%s ' --packed-cache ) \
  --max-chunks "$MAX_CHUNKS"
