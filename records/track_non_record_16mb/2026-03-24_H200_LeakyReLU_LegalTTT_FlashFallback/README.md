# LeakyReLU² + Legal Score-First TTT + Parallel Muon

**val_bpb: 1.1194** (3-seed mean, std 0.0006) | **~15.95 MB** | 8×H100 SXM

## H200 Status

This folder is our ICRN H200 reproduction and submission-prep copy of the public March 23 record stack. It is **not** itself a submit-ready record package, but it now contains a strong local recovery of the lane:

| Hardware | Lane | Metric | Value | Bytes | Notes |
|----------|------|--------|-------|-------|-------|
| 1×H200 NVL | Exact 80-shard run | `final_int6_sliding_window_exact` | `1.11623907` | `15,860,692` | Full train + quant + sliding eval completed |
| 1×H200 NVL | Resumed legal TTT on saved artifact | `legal_ttt_exact` | `1.11382777` | `15,860,692` | TTT completed in a resumed eval-only pass |

The `1.11382777` result is leaderboard-worthy numerically against the public `1.1194` score, but it is **not yet a valid submission candidate on the organizer-approved H200 path** because the underlying training run took `7,503,164 ms`, which is well over the `4,615,816 ms` H200-equivalent train cap. Unless organizers explicitly waive it, significance evidence would still be needed for a new SOTA claim.

For current H200 development, we now treat the empirical proxy for the `8xH100` 600-second train cap as a hard dev-side guardrail:

- `<= 7,185` steps
- `<= 4,615,816 ms` on `1xH200 NVL` (`~76.9 min`)

The H200 launchers in `scripts/` default to this proxy budget unless `ALLOW_OUT_OF_BUDGET_DEV_RUN=1` is set intentionally.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 83.3ms | 7,179 | 1.1217 | **1.1192** | -0.0025 | 410s | 15,977,386 |
| 42 | 83.4ms | 7,182 | 1.1227 | **1.1200** | -0.0027 | 408s | 15,876,510 |
| 2025 | 83.4ms | 7,193 | 1.1212 | **1.1189** | -0.0023 | 408s | 15,990,006 |
| **Mean** | **83.4ms** | **7,185** | **1.1218** | **1.1194 (std 0.0006)** | **-0.0025** | **~409s** | |

## Key Innovation: LeakyReLU(0.5)²

One-line activation change that delivers -0.003 BPB:

```python
# Standard (relu²)
x = torch.relu(self.fc(x)).square()

# This submission (leaky relu²)
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

LeakyReLU with slope 0.5 preserves negative gradient flow through the MLP, allowing the model to learn from both positive and negative pre-activations. The squaring step still produces non-negative outputs, maintaining the relu² inductive bias while eliminating dead neurons.

This activation is used in PR #493 (ablated at -0.003 BPB) and PR #518 (part of their 1.0622 record submission).

## Legal TTT Protocol

Backward-looking, score-first TTT following PR #461's framework:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation possible
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

`inference_mode()` is a PyTorch context manager that disables gradient tracking and prohibits in-place weight mutation, providing a hard guarantee that scoring is stateless.

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |

### Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (≤10 min) |
| Standard eval (int6 roundtrip + sliding window) | ~120s |
| Legal TTT (score-first sliding + adaptation) | ~410s |
| **Total eval** | **~530s (< 10 min)** |

## Training Architecture

PR #414 stack with Parameter Banking + Parallel Muon (PR #399):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

### Parameter Banking + Parallel Muon

First introduced in [PR #399](https://github.com/openai/parameter-golf/pull/399):

- 4 contiguous 3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights
- Batched Newton-Schulz orthogonalization via `torch.bmm`
- DDP removed for banks; async reduce-scatter → local NS → async all-gather
- 83.3ms/step vs ~85ms baseline

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Repo Launch Helpers

- H200 proxy reproduction: `scripts/icrn_h200_ttt_recordstack.sh`
- H200 resumed legal-TTT recovery: `scripts/icrn_h200_resume_legal_ttt.sh`
- H200 artifact-only legal-TTT candidate launcher: `scripts/icrn_h200_artifact_ttt_candidate.sh`
- H200 artifact-only legal-TTT portfolio helper: `scripts/icrn_h200_artifact_ttt_portfolio.sh`
- H200 H100-step proxy: `scripts/icrn_h200_ttt_h100_proxy.sh`
- H200 generic proxy candidate runner: `scripts/icrn_h200_ttt_h100_proxy_candidate.sh`
- H200-first record-push orchestrator: `scripts/icrn_h200_record_push.sh`
- H200 search-state reporter + `8xH100` handoff helper: `scripts/record_push_status.py`
- 8xH100 repro/submit path: `scripts/h100_repro_leaky_ttt_parallel_muon.sh`
- 8xH100 3-seed wrapper: `scripts/h100_repro_leaky_ttt_parallel_muon_3seed.sh`
- 8xH100 generic promoted-candidate runner: `scripts/h100_record_push_candidate.sh`
- 8xH100 generic promoted-candidate 3-seed wrapper: `scripts/h100_record_push_candidate_3seed.sh`
- 8xH100 parallel portfolio selector: `scripts/h100_parallel_candidate_portfolio.sh`
- 8xH100 parallel portfolio 3-seed wrapper: `scripts/h100_parallel_candidate_3seed.sh`
- Log-to-submission metadata: `scripts/prepare_submission_metadata.py`
- Multi-run summary helper: `scripts/summarize_record_runs.py`

## H200-First Push Order

Use the encoded search order instead of ad hoc launches:

```bash
bash scripts/icrn_h200_record_push.sh artifact
bash scripts/icrn_h200_record_push.sh proxy
bash scripts/icrn_h200_record_push.sh combined
bash scripts/icrn_h200_record_push.sh report
```

or run the whole sequence in one command:

```bash
bash scripts/icrn_h200_record_push.sh all
```

The orchestrator uses this exact order:

1. Artifact-only legal-TTT sweep:
   `baseline -> tttlr25 -> batch48 -> tttlr25_batch48 -> bg3072_tttlr25 -> chunk16k -> epochs2_tttlr25 -> freeze2_tttlr25 -> freeze2_epochs2_tttlr25 -> tttlr30`
2. H200 H100-step proxy sweep:
   `baseline -> vr1 -> bg3072 -> vr1_bg3072`
3. One combined H200 proxy run only if both the best non-baseline artifact candidate and best non-baseline proxy candidate exist.
4. Promote exactly one winner and keep exactly one fallback for `8xH100`.

At any point you can inspect the current rankings and exact `8xH100` handoff commands with:

```bash
python scripts/record_push_status.py --seed 1337
```

## Parallel H100 Portfolio

When `8xH100` access is available, the fastest way to search beyond the recovered winner is to branch only on the highest-signal remaining ambiguities:

- `baseline`: recovered winning stack
- `vr1`: add `VALUE_RESIDUAL=1`
- `bg3072`: raise `BIGRAM_VOCAB_SIZE` from `1536` to `3072`
- `vr1_bg3072`: combine the two architecture changes
- `tttlr25`: keep training fixed and increase `TTT_LR` from `0.002` to `0.0025`
- `vr1_bg3072_tttlr25`: combo bet on all three knobs

Each candidate has its own launcher under `scripts/`, and `scripts/h100_parallel_candidate_portfolio.sh` prints the one-command invocations for spreading them across multiple `8xH100` nodes in parallel.

For the promoted-candidate handoff path used by the H200-first push, use:

```bash
ARCH_CANDIDATE=baseline TTT_CANDIDATE=baseline SEED=1337 \
bash scripts/h100_record_push_candidate.sh
```

and only after the exact `8xH100` seed-1337 run clears bytes, train/eval time, and `legal_ttt_exact <= 1.1144`, continue with:

```bash
ARCH_CANDIDATE=baseline TTT_CANDIDATE=baseline SEEDS="1337 42 2025" \
bash scripts/h100_record_push_candidate_3seed.sh
```

You can package the recovered winning H200 lane with:

```bash
python scripts/prepare_submission_metadata.py \
  records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs/h200_ttt_recordstack_80shard_seed1337.txt \
  records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs/h200_ttt_recordstack_80shard_seed1337_resume_ttt.txt
```

or summarize it as one logical run with:

```bash
python scripts/summarize_record_runs.py --merge-logs \
  records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs/h200_ttt_recordstack_80shard_seed1337.txt \
  records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/logs/h200_ttt_recordstack_80shard_seed1337_resume_ttt.txt
```

The H200 copy of `train_gpt.py` in this folder is source-equivalent to the public record stack except for a safe attention fallback: if `flash_attn_interface` is installed it uses the original FA3 path, otherwise it falls back to PyTorch SDPA with GQA enabled. That lets us validate the exact stack on a single H200 now and later reuse the same folder on H100s without reconstructing the setup from scratch.

## Ablation

Incremental contribution of each technique (all seed 1337):

| Change | Pre-TTT bpb | Post-TTT bpb | Delta |
|--------|-------------|-------------|-------|
| PR #414 base (relu², BIGRAM=2048) | 1.1234 | — | — |
| + Parameter Banking + Parallel Muon | 1.1234 | — | ±0.0000 |
| + Legal TTT (3ep, freeze=2) | — | 1.1217 | -0.0017 |
| + TTT freeze=0 (all blocks) | — | 1.1213 | -0.0004 |
| + BigramHash 2048→3072 | — | 1.1204 | -0.0009 |
| + **LeakyReLU(0.5)²** | 1.1213 | **1.1183** | **-0.0021** |

## Credits

- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon (adapted: freeze=0 instead of original freeze=2)
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
