# Rank-1 Mixed-QAT Warmdown-Ramp Variant

Derived from the official March 20 record snapshot:

- source seed: `workbench/official_top_records/rank_01_2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`
- base score: `1.1428` mean val_bpb across 3 seeds

This variant keeps the rank-1 10-layer mixed int5/int6 architecture and adds a training-time fake-quant path on the matrices that are quantized at export:

- block MLP weights use int5-style STE fake quantization
- attention matrices use int6-style STE fake quantization
- the BigramHash projection uses int6-style STE fake quantization
- QAT strength ramps in during warmdown, reaching full strength before the rank-1 SWA collection window (`lr_scale` from `0.8` down to `0.4`)

The goal is to preserve the official #1 script's early-training dynamics, then spend the converged warmdown window adapting weights toward the mixed int5/int6 roundtrip that actually determines record-track quality.

## Run

```bash
python3 scripts/run_record_experiment.py \
  --experiment-dir record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_warmdown_ramp_seed42_20260323 \
  --seed 42 \
  --nproc-per-node 8 \
  --stats-path runs/rank1_mixed_qat_warmdown_ramp_seed42_20260323/stats.json
```

## Colab Pilot

For a smaller single-GPU Colab pilot run:

```bash
python3 scripts/run_record_experiment.py \
  --experiment-dir record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_colab_pilot_seed42 \
  --seed 42 \
  --nproc-per-node 1 \
  --required-cuda-devices 1 \
  --set-env-file record_candidates/2026-03-23_rank1_mixed_qat/colab_pilot_env.json \
  --stats-path runs/rank1_mixed_qat_colab_pilot_seed42/stats.json
```
