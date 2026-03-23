# Rank-1 Mixed-QAT Variant

Derived from the official March 20 record snapshot:

- source seed: `workbench/official_top_records/rank_01_2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`
- base score: `1.1428` mean val_bpb across 3 seeds

This variant keeps the rank-1 10-layer mixed int5/int6 architecture and adds a training-time fake-quant path on the matrices that are quantized at export:

- block MLP weights use int5-style STE fake quantization
- attention matrices use int6-style STE fake quantization
- the BigramHash projection uses int6-style STE fake quantization

The goal is to reduce the post-export roundtrip penalty on the current best official script without reverting to the older generic baseline family.

## Run

```bash
python3 scripts/run_record_experiment.py \
  --experiment-dir record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_seed42_20260323 \
  --seed 42 \
  --nproc-per-node 8 \
  --stats-path runs/rank1_mixed_qat_seed42_20260323/stats.json
```
