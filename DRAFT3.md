# DRAFT3

Prepared: March 19, 2026

This is the current reviewer packet after the work recorded in `DRAFT2.md`.

`DRAFT2.md` ended with the `24k` exact branch at `1.6238459688827054` and a live `32k` tokenizer run that had only reached `step=800`.
This file captures what happened next and why the search direction is now slightly different.

## 1. Executive Snapshot

Current facts that matter most:

- The official leaderboard is unchanged as of March 19, 2026.
  - main-track score to beat: `1.22436570`
  - practical clean record target after the `0.005 nats` margin: `1.217152224795555`
- Our best completed local exact result is still:
  - `bytelevel24k_d512_gqa_softcap_s3200`
  - `final_val_bpb = 1.6238459688827054`
- The `32k` tokenizer branch was stopped early at:
  - `bytelevel32k_d512_gqa_softcap_s3200`
  - `step=2400 val_bpb = 1.6675`
- The active frontier is now:
  - `bytelevel48k_d512_gqa_softcap_s3200`
  - `step=0 val_bpb = 3.5214`
- The `32k` branch is no longer better than `24k`.
  - at `step=800`, `32k` was ahead
  - at `step=1600`, `32k` was behind `24k`
  - at `step=2400`, `32k` was still behind `24k` (`1.6675` versus `1.6536`)
- A new model-side branch is now ready:
  - `MLP_KIND=relu2`
  - `USE_BLOCK_SCALES=1`
  - `USE_RESID_MIX=1`
  - smoke-tested successfully, not yet run in a long exact experiment

Main conclusion:

- The project is no longer blocked on infrastructure.
- The current decision is no longer “wait for `32k` to rescue us.”
- We have already promoted `48k`, and the next real decision is whether tokenizer scaling still has enough headroom before we switch to the new baseline-inspired block variant.

## 2. Official Score Check

The official repo still shows the same March 18 records when rechecked on March 19:

- `Naive Baseline`: `1.22436570`
- `4-Hour Baseline` non-record: `1.20737944`

This matters because nothing in our local search is PR-ready yet.

Our best local exact result is still:

```text
1.6238459688827054
```

Gap to the official main-track leader:

```text
1.6238459688827054 - 1.22436570 = 0.3994802688827055 bpb
```

Gap to the local interim target:

```text
1.6238459688827054 - 1.5 = 0.12384596888270538 bpb
```

## 3. What Changed Since `DRAFT2`

### 3.1 The `32k` tokenizer run stopped looking like an obvious win and was cut early

At the time of `DRAFT2`, the best live checkpoint was:

```text
step=800 val_bpb=1.7785
```

That looked encouraging because the `24k` branch was `1.8085` at the same step.

The run later advanced to:

```text
step=1600 train_loss=4.7604 train_bpb=1.5083 val_loss=5.1868 val_bpb=1.6975
step=2400 train_loss=4.5696 train_bpb=1.4266 val_loss=5.0579 val_bpb=1.6675
```

That changes the interpretation:

- `32k` kept improving
- but it never got back ahead of the `24k` branch at matched horizon
- it would have needed an unrealistic final-segment drop to hit `1.5`
- so it was stopped after `step=2400` and replaced immediately by `48k`

This is important because it means “bigger vocab” is still promising, but the `32k` rung itself was not enough.

### 3.2 We added a new baseline-inspired model branch

`train_gpt.py` now supports three new controls:

```python
mlp_kind=os.environ.get("MLP_KIND", "swiglu").strip().lower()
use_block_scales=env_bool("USE_BLOCK_SCALES", False)
use_resid_mix=env_bool("USE_RESID_MIX", False)
```

The block now supports `ReLU^2` in addition to `SwiGLU`:

```python
if self.mlp_kind == "swiglu":
    return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))
hidden = torch.relu(self.ff_up(x))
return self.ff_down(hidden * hidden)
```

It also supports learned block output scaling:

```python
if self.attn_scale is not None:
    attn_out = attn_out * self.attn_scale.to(dtype=attn_out.dtype).view(1, 1, -1)
...
if self.mlp_scale is not None:
    mlp_out = mlp_out * self.mlp_scale.to(dtype=mlp_out.dtype).view(1, 1, -1)
```

And a learned residual mix against the loop input:

```python
if self.resid_mix is not None:
    mix = self.resid_mix.to(dtype=x.dtype)
    x = mix[0].view(1, 1, -1) * x + mix[1].view(1, 1, -1) * x0
```

Why this matters:

- it gives us a new model-side lever that is closer to the official baseline family
- it does not require deleting the current looped/GQA/softcap path
- it is ready to combine with either the `24k` or `48k` tokenizer branch if the pure denominator climb stalls

## 4. Exact Commands And Outputs

### 4.1 `32k` checkpoint inspection before promotion

Command:

```bash
tail -n 80 runs/bytelevel32k/bytelevel32k_d512_gqa_softcap_s3200.log
```

Output:

```text
step=0 train_loss=10.4906 train_bpb=3.3573 val_loss=10.4514 val_bpb=3.4403 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=800 train_loss=5.0753 train_bpb=1.6506 val_loss=5.4205 val_bpb=1.7785 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1279.1s
step=1600 train_loss=4.7604 train_bpb=1.5083 val_loss=5.1868 val_bpb=1.6975 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=2624.8s
step=2400 train_loss=4.5696 train_bpb=1.4266 val_loss=5.0579 val_bpb=1.6675 muon_lr=4.351e-03 adamw_lr=6.527e-05 elapsed=4055.2s
```

Interpretation:

- The run was healthy but no longer the best use of MPS time.
- It was still slower than the `24k` branch in bpb terms at matched step.
- That was enough evidence to stop it early and promote `48k`.

### 4.2 Launch the `48k` tokenizer branch

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d512_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=800 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d512_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d512_gqa_softcap_s3200.log
```

Output:

```text
vocab_size=49152 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_48k_sample/val
avg_bytes_per_token=4.4668 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_48k_sample/val
parameters=28,053,128
step=0 train_loss=10.8960 train_bpb=3.4711 val_loss=10.8959 val_bpb=3.5214 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

Interpretation:

- `48k` is now the active frontier.
- This is the largest denominator step taken so far without jumping all the way to `64k`.
- If `48k` still cannot beat the `24k` branch by `step=800`, that will be a strong signal that denominator scaling alone is flattening and the new block branch should move up in priority.

### 4.3 Smoke test for the new block knobs

Command:

```bash
RUN_ID=baseline_block_smoke DEVICE=cpu MAX_STEPS=1 VAL_LOSS_EVERY=1 VAL_STEPS=1 D_MODEL=64 N_HEADS=8 NUM_KV_HEADS=4 D_FF=128 N_LOOPS=2 SEQ_LEN=32 TRAIN_BATCH_TOKENS=256 VAL_BATCH_TOKENS=256 VOCAB_SIZE=512 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=3.5 TIED_EMBEDDINGS=1 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 MLP_KIND=relu2 USE_BLOCK_SCALES=1 USE_RESID_MIX=1 uv run python train_gpt.py
```

Output:

```text
parameters=62,104
step=0 train_loss=6.2520 train_bpb=2.5771 val_loss=6.2495 val_bpb=2.5760 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=0.11
final_val_bpb=2.5767
total_artifact_bytes=131827
artifact_budget_ok=True
```

Interpretation:

- The new branch is real, not just drafted.
- It is safe to launch as soon as the current MPS run frees up.
- The active `32k` run does not include these changes, because it started before this code landed.

## 5. What I Think Now

The search tree is now:

1. Let the live `48k` tokenizer branch reach at least `step=800`.
2. If it still trails the `24k` branch badly, switch priority to the new `relu2 + block_scales + resid_mix` model branch.
3. If `48k` looks close but still short, keep `64k` as the next denominator contingency.

Why `48k` and not immediately `64k`:

- `48k` gives another meaningful bytes/token increase
- it is less likely than `64k` to collapse optimization by spending too much capacity on the embedding table
- it is the cleaner next denominator test

Why the new block branch matters:

- the `24k` `3200`-step branch only improved `1.6664 -> 1.6238` in its last segment
- the `32k` branch is not clearly beating it
- that is the shape of a search frontier that may need another model-side lever, not only another tokenizer rung

## 6. Reviewer Questions

These are the questions I would most want external feedback on now:

1. Is `48k` the right next tokenizer rung, or should the project skip directly to `64k` given the remaining `0.1238` gap to `1.5`?
2. Are the new `relu2 + block_scales + resid_mix` knobs directionally sensible, or is there a better baseline-inspired approximation to try first?
3. Is the current `d_model=512, n_loops=4, GQA, softcap` recipe already too embedding-heavy for vocabularies above `24k`?
4. Should the next big push be denominator-first (`48k`) or nats-first (new block branch on `24k`)?

## 7. Current Bottom Line

The best completed local exact result is still:

```text
1.6238459688827054
```

The most important stopped comparison run is:

```text
bytelevel32k_d512_gqa_softcap_s3200
step=2400 val_bpb=1.6675
```

The active frontier is:

```text
bytelevel48k_d512_gqa_softcap_s3200
step=0 val_bpb=3.5214
```

The next prepared model-side branch is:

```text
MLP_KIND=relu2
USE_BLOCK_SCALES=1
USE_RESID_MIX=1
```

We are still not at `1.5`, but the project now has both of the remaining obvious levers active or prepared:

- larger-tokenizer denominator scaling beyond `24k`
- a new model-side branch beyond the current GQA/softcap recipe
