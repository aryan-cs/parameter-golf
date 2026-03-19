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
  - `bytelevel24k_d640_gqa_softcap_s3200`
  - `step=1600 val_bpb = 1.6102`
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
- We promoted `48k`, rejected the full-width local form for memory reasons, and even the healthier accumulated `48k d384` path still lost by `step=1600`.
- The `relu2` block family underperformed, so the live question is now whether a wider plain `24k` model is the simpler path to lower bpb.
- The new `step=1600` width checkpoint says “yes, at least locally”: `d640` is now `0.0562` bpb better than the old `d512` curve at matched horizon.

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

### 3.2 The first `48k` form was wrong, but the rung is still alive

The first attempt was:

```text
bytelevel48k_d512_gqa_softcap_s3200
```

It initialized and reached `step=0`, but system inspection showed it was a bad local search configuration:

```text
PID   COMMAND    %CPU MEM TIME     #TH  STATE
1972  python3.13 0.0  27G 03:11.47 13/1 running
```

and:

```text
PhysMem: 23G used (9122M wired, 3518M compressor), 59M unused.
VM: 185T vsize, 5268M framework vsize, 179957826(0) swapins, 194681707(0) swapouts.
```

So the project did not abandon `48k`; it changed the local form:

- resize from `d512` to `d384`
- keep the effective batch at `16384`
- lower peak memory with `TRAIN_MICROBATCH_TOKENS=8192`

That branch later reached:

```text
step=1600 train_loss=4.8752 train_bpb=1.5162 val_loss=5.4769 val_bpb=1.7439
```

The relevant older `24k` reference at the same checkpoint was:

```text
step=1600 ... val_bpb=1.6664
```

So the resized `48k` branch was still `0.0775` bpb worse, which is well outside the local proxy noise band.

That is why the project stopped denominator scaling as the main hypothesis and moved back to `24k`-first model search.

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

### 3.3 The block-variant branch is now the live test

The current active run is:

```text
bytelevel24k_d512_gqa_softcap_relu2_s1600
```

Its first real checkpoint is:

```text
step=400 train_loss=5.5333 train_bpb=1.8787 val_loss=5.7186 val_bpb=1.9406
```

The older plain `24k` reference at `step=400` was:

```text
step=400 ... val_bpb=1.9226
```

So the new block branch starts `0.0180` bpb worse than the old one.

That is not a clear win, but it is still inside the local `+/- 0.05` proxy-noise band, so this branch deserves at least one more segment before being rejected.

At the next checkpoint it reached:

```text
step=800 train_loss=5.3204 train_bpb=1.7638 val_loss=5.4004 val_bpb=1.8192
```

The closest old `24k` reference with the same `1600`-step schedule was:

```text
step=800 ... val_bpb=1.7954
```

So the new branch is `0.0238` bpb worse at `step=800`.

That is still inside the local proxy-noise band, so the branch has not failed yet, but it also has not earned the right to be called better.

The simplified `relu2`-only follow-up behaved almost the same way:

```text
step=800 train_loss=5.3235 train_bpb=1.7648 val_loss=5.3960 val_bpb=1.8177
```

That is still `0.0223` bpb worse than the old plain `24k` reference (`1.7954`).

So the project stopped spending MPS time on `relu2` variants and moved to the next cleaner lever: more width on the plain winning recipe.

### 3.4 A wider plain `24k` branch is now the strongest live hypothesis

The current active run is:

```text
bytelevel24k_d640_gqa_softcap_s1600
```

Its first real checkpoint is:

```text
step=400 train_loss=5.3712 train_bpb=1.8237 val_loss=5.5845 val_bpb=1.8951
step=800 train_loss=5.0739 train_bpb=1.6821 val_loss=5.2047 val_bpb=1.7533
```

The older plain `24k` `d512` reference at the same checkpoint was:

```text
step=400 ... val_bpb=1.9226
step=800 ... val_bpb=1.7954
```

So the wider branch is:

- `0.0275` bpb better at `step=400`
- `0.0421` bpb better at `step=800`

The wider branch then continued to:

```text
step=1200 train_loss=4.6817 train_bpb=1.6219 val_loss=5.0355 val_bpb=1.7210
=== final_stats ===
steps=1600
final_val_bpb=1.6874
```

Relative to the older plain `24k d512` reference, that means:

- `0.0275` bpb better at `step=400`
- `0.0421` bpb better at `step=800`
- `0.0546` bpb better at `step=1200`
- `0.0610` bpb better at the completed `1600`-step finish

This is the first branch in a while that clearly beats the old reference over a full local run, which is why the project immediately promoted it to:

```text
bytelevel24k_d640_gqa_softcap_s3200
```

The promoted continuation has started cleanly:

```text
step=0 val_bpb=3.4676
step=800 val_bpb=1.7610
```

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

### 4.2 Launch the first `48k` tokenizer branch

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

- This proved the denominator rung existed, but not that the local shape was usable.
- The full-width version was rejected for memory reasons.

### 4.3 Resize and accumulate the `48k` branch

Reference parameter counts:

```text
24576 512 1365 15470216
32768 512 1365 19664520
49152 384 1024 20499560
49152 320 853 16857368
```

The direct resized probe:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d384_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_s3200.log
```

Output before interrupt:

```text
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4639 val_loss=10.8440 val_bpb=3.5047 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

The microbatch smoke:

```bash
RUN_ID=bytelevel48k_d384_gqa_softcap_accum_smoke TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 TRAIN_MICROBATCH_TOKENS=8192 VAL_BATCH_TOKENS=8192 VAL_STEPS=1 VAL_LOSS_EVERY=1 MAX_STEPS=1 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_smoke.json uv run python train_gpt.py
```

Output:

```text
train_batch_size_tokens=16384 train_microbatch_size_tokens=8192 grad_accum_steps=2
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4118 val_loss=10.4851 val_bpb=3.5006 muon_lr=4.000e-02 adamw_lr=6.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=24.29
final_val_bpb=3.3818
```

The real accumulated launch:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d384_gqa_softcap_accum_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 TRAIN_MICROBATCH_TOKENS=8192 VAL_BATCH_TOKENS=8192 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_s3200.log
```

Output so far:

```text
train_batch_size_tokens=16384 train_microbatch_size_tokens=8192 grad_accum_steps=2
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4118 val_loss=10.8438 val_bpb=3.4832 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=400 train_loss=5.8386 train_bpb=1.8479 val_loss=5.9844 val_bpb=1.9461 muon_lr=3.805e-02 adamw_lr=5.707e-04 elapsed=550.5s
step=800 train_loss=5.1903 train_bpb=1.6323 val_loss=5.6920 val_bpb=1.8215 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1099.6s
step=1200 train_loss=5.2098 train_bpb=1.6498 val_loss=5.5890 val_bpb=1.7978 muon_lr=2.400e-02 adamw_lr=3.600e-04 elapsed=1648.3s
```

Interpretation:

- This is the first `48k` branch shape that both keeps the `48k` tokenizer denominator and behaves like a sane local training job.
- It is now the correct active frontier.
- Relative to the older `24k` `step=400` checkpoint (`1.9226`), it was only `0.0235` bpb worse.
- By `step=800`, it is only `0.0130` bpb worse than the older `24k` checkpoint (`1.8085`).
- The gap narrowed because the `48k` branch improved faster over the `400 -> 800` segment.
- By `step=1200`, it is `0.0222` bpb worse than the older `24k` checkpoint (`1.7756`).
- So this branch is still alive, but it now needs a stronger `1200 -> 1600` segment to stay ahead of the queued model-side ablation.

### 4.4 Smoke test for the new block knobs

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

### 3.7 The `d640` continuation became the best live branch by a meaningful margin

The current live command is:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=800 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s3200.log
```

The run already looked encouraging at `step=800`:

```text
step=800 train_loss=5.0383 train_bpb=1.6703 val_loss=5.1894 val_bpb=1.7610 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1189.7s
```

The more important checkpoint has now landed:

```text
step=1600 train_loss=4.4627 train_bpb=1.5096 val_loss=4.7798 val_bpb=1.6102 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=2379.6s
```

Why this matters:

- the old plain `24k d512` `3200`-step curve was `1.6664` at `step=1600`
- the live `d640` continuation is `1.6102` at the same horizon
- that is a real matched-horizon gain of `0.0562` bpb
- the gap from the current live checkpoint to the local `1.5` target is now `0.1102` bpb

This changes the recommendation:

- do not cut this run early
- let it continue through `step=2400` and `step=3200`
- if it later stalls, width bracketing (`d576` or `d704`) is still the cleanest next move
- do not fall back to `48k` or `relu2` first, because width is now the first branch that has held its advantage at matched horizon

## 5. What I Think Now

The search tree is now:

1. Let the live `bytelevel24k_d640_gqa_softcap_s3200` run continue through `step=2400` and `step=3200`.
2. If it holds most of its midpoint edge, keep pushing width in this neighborhood rather than jumping tokenizers again.
3. If it flattens hard from here, bracket the width sweet spot with `d576` and `d704`.
4. Keep the `relu2 + block_scales + resid_mix` branch as a prepared fallback, not the first next move.
5. Treat `48k` and `64k` as contingency denominator branches only after the width line stops paying.

Why width is now the lead hypothesis:

- `d640` is now `0.0562` bpb better than the old plain `d512` curve at matched `step=1600`
- that is outside the local proxy noise band we have been using
- the gain survived long enough that it no longer looks like a warmup artifact
- width is the first recent lever that stayed ahead after a fair matched-horizon check

Why I am not prioritizing `48k` or `relu2` first:

- the healthiest `48k` local branch still lost clearly by `step=1600`
- the `relu2` family underperformed both in the all-at-once form and in the simpler core-only form
- the current live `d640` branch is closer to `1.5` than either of those branches ever got at the same maturity

## 6. Reviewer Questions

These are the questions I would most want external feedback on now:

1. If the live `d640` branch finishes just above `1.5`, is the next best move `d704`, or is that likely to become a memory/optimization regression on this hardware?
2. Is there a better way to spend extra capacity in the `24k` family than pure width, given that tokenizer scaling beyond `24k` has not yet paid off locally?
3. Are there baseline-inspired tweaks more promising than the current `relu2 + block_scales + resid_mix` branch, which now looks second-tier?
4. How much confidence should we place in a `0.0562` bpb matched-horizon gain on this local setup before committing more wall-clock to width bracketing?

## 7. Current Bottom Line

The best completed local exact result is still:

```text
1.6238459688827054
```

The most important stopped comparison run is:

```text
bytelevel48k_d384_gqa_softcap_accum_s3200
step=1600 val_bpb=1.7439
```

The active frontier is:

```text
bytelevel24k_d640_gqa_softcap_s3200
step=1600 val_bpb=1.6102
```

The prepared model-side ablation is:

```text
MLP_KIND=relu2
USE_BLOCK_SCALES=1
USE_RESID_MIX=1
```

We are still not at `1.5`, but the project now has both of the remaining obvious levers active or prepared:

- larger-tokenizer denominator scaling beyond `24k`
- a new model-side branch beyond the current GQA/softcap recipe
