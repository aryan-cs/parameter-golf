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
  - `bytelevel24k_d640_gqa_softcap_cd05_s3200`
  - `final_val_bpb = 1.6017072436714903`
- The `32k` tokenizer branch was stopped early at:
  - `bytelevel32k_d512_gqa_softcap_s3200`
  - `step=2400 val_bpb = 1.6675`
- The most recent stopped width continuation is now:
  - `bytelevel24k_d640_gqa_softcap_cd05_s4800`
  - `step=3200 val_bpb = 1.6063`
- The most recent stopped schedule-retune branch is now:
  - `bytelevel24k_d640_gqa_softcap_cd10_s4800`
  - `step=1600 val_bpb = 1.6162`
- The active frontier is now:
  - `bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800`
  - `step=0 val_bpb = 3.4671`
- The prepared serial fallback queue is now:
  - `bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800`
  - `bytelevel24k_d576_gqa_softcap_cd05_s4800`
  - launched via `queue_runs.py` and `queues/d640_followups.json`
- The prepared post-queue knob sweep is now:
  - `bytelevel24k_d640_gqa_softcap_cd08_w100_s3200`
  - `bytelevel24k_d640_gqa_qg175_cd05_s1600`
  - `bytelevel24k_d640_gqa_softcap_nosmear_cd05_s1600`
  - launched via `queue_runs.py` and `queues/d640_knob_queue.json`
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
- The width answer is now clearer: `d640` stayed ahead through `step=2400`, but the original schedule flattened hard enough that the next live test is a later-cooldown restart of the same width recipe.

## 2. Official Score Check

The official repo still shows the same March 18 records when rechecked on March 19:

- `Naive Baseline`: `1.22436570`
- `4-Hour Baseline` non-record: `1.20737944`

This matters because nothing in our local search is PR-ready yet.

Our best local exact result is now:

```text
1.6017072436714903
```

Gap to the official main-track leader:

```text
1.6017072436714903 - 1.22436570 = 0.3773415436714904 bpb
```

Gap to the local interim target:

```text
1.6017072436714903 - 1.5 = 0.10170724367149031 bpb
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

### 3.7 The `d640` continuation became the best width signal, then exposed a schedule problem

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

The next checkpoint then changed the decision:

```text
step=2400 train_loss=4.2979 train_bpb=1.3887 val_loss=4.7161 val_bpb=1.6118 muon_lr=4.351e-03 adamw_lr=6.527e-05 elapsed=3569.9s
```

That is still better than the old plain `24k d512` curve at the same horizon:

```text
1.6536 -> 1.6118
```

But it is not better than the same run at `step=1600`:

```text
1.6102 -> 1.6118
```

That matters a lot.

- width is still the best confirmed direction
- but the original `COOLDOWN_FRACTION=0.20` schedule is now a real suspect
- by `step=2400`, `muon_lr` had already fallen to `4.351e-03`
- the branch no longer looked like a plausible direct path to `1.5` on that exact schedule

So the project stopped that run early on purpose and immediately relaunched the same `d640` recipe with later decay:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_cd05_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd05_s3200.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd05_s3200.log
```

Launch output:

```text
parameters=20,238,248
step=0 train_loss=10.1780 train_bpb=3.3062 val_loss=10.1269 val_bpb=3.4676 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

The first real checkpoint from the longer-horizon retry is:

```text
step=400 train_loss=5.3749 train_bpb=1.8249 val_loss=5.5847 val_bpb=1.8951 muon_lr=3.938e-02 adamw_lr=5.907e-04 elapsed=599.2s
```

Compared with the completed `3200`-step `d640 + cd05` branch:

```text
3200-step branch at step=400: 1.8955
4800-step branch at step=400: 1.8951
```

That is only a `0.0004` bpb edge, so it is basically a tie, but at least the longer run is not losing early.

The next checkpoint keeps the same story alive:

```text
step=800 train_loss=5.0293 train_bpb=1.6673 val_loss=5.1540 val_bpb=1.7362 muon_lr=3.744e-02 adamw_lr=5.616e-04 elapsed=1212.4s
```

Compared with the completed `3200`-step branch:

```text
3200-step branch at step=800: 1.7390
4800-step branch at step=800: 1.7362
```

That is a `0.0028` bpb gain.

It is not dramatic, but it is enough to keep the longer-horizon branch alive for the later checkpoints where extra optimization room should matter more.

The next two checkpoints narrow that story:

```text
step=1200 train_loss=4.5526 train_bpb=1.5772 val_loss=4.9028 val_bpb=1.6756 muon_lr=3.433e-02 adamw_lr=5.149e-04 elapsed=1949.5s
step=1600 train_loss=4.4452 train_bpb=1.5036 val_loss=4.7855 val_bpb=1.6169 muon_lr=3.027e-02 adamw_lr=4.541e-04 elapsed=2834.9s
```

Compared with the completed `3200`-step branch:

```text
3200-step branch at step=1200: 1.6760
4800-step branch at step=1200: 1.6756

3200-step branch at step=1600: 1.6143
4800-step branch at step=1600: 1.6169
```

So the longer-horizon retry is:

```text
0.0004 bpb better at step=1200
0.0026 bpb worse at step=1600
```

That is basically a tie, not a failure.

- the branch is still close enough to matter
- the whole point of the longer horizon is the later segment, not the midpoint
- and it is still carrying much higher learning rates than the `3200`-step run at this point

So it stays alive for the `step=2000` and `step=2400` comparisons.

The next checkpoint still does not break the tie:

```text
step=2000 train_loss=4.1914 train_bpb=1.3970 val_loss=4.7615 val_bpb=1.6054 muon_lr=2.559e-02 adamw_lr=3.838e-04 elapsed=3801.9s
```

Compared with the completed `3200`-step branch:

```text
3200-step branch at step=2000: 1.6028
4800-step branch at step=2000: 1.6054
```

That is only `0.0026` bpb worse.

So the longer-horizon retry is still unresolved, not invalidated. The next real verdict is `step=2400`, where the completed `3200`-step run hit `1.5943`.

That comparison has now landed:

```text
step=2400 train_loss=4.2118 train_bpb=1.3609 val_loss=4.6866 val_bpb=1.5969 muon_lr=2.063e-02 adamw_lr=3.095e-04 elapsed=4760.1s
```

Compared with the completed `3200`-step branch:

```text
3200-step branch at step=2400: 1.5943
4800-step branch at step=2400: 1.5969
```

That is `0.0026` bpb worse, which is still basically a tie.

So the longer-horizon retry is still alive. It has not proven that extra horizon is better yet, but it also has not lost by enough to justify cutting it before the `3200` checkpoint.

The first real checkpoint from that restart is now:

```text
step=400 train_loss=5.3774 train_bpb=1.8258 val_loss=5.5858 val_bpb=1.8955 muon_lr=3.861e-02 adamw_lr=5.792e-04 elapsed=599.0s
```

That is early, but it is already ahead of the old plain `24k d512` `step=400` checkpoint:

```text
1.9226 -> 1.8955
```

So the later-cooldown retry is alive enough to keep running.

The next checkpoint is even more important because it is finally apples-to-apples against the original `d640` schedule:

```text
step=800 train_loss=5.0390 train_bpb=1.6705 val_loss=5.1623 val_bpb=1.7390 muon_lr=3.439e-02 adamw_lr=5.159e-04 elapsed=1198.1s
```

Comparison:

```text
original d640 schedule at step=800: 1.7610
late-cooldown d640 at step=800:     1.7390
```

That is a real gain of:

```text
0.0220 bpb
```

So the later-cooldown retry is no longer just a reasonable restart. It is now a better branch than the original `d640` run at the first matched checkpoint.

The next live checkpoint continues the same direction:

```text
step=1200 train_loss=4.5453 train_bpb=1.5746 val_loss=4.9039 val_bpb=1.6760 muon_lr=2.806e-02 adamw_lr=4.209e-04 elapsed=1798.5s
```

That matters because:

- the restarted branch is still improving meaningfully after `step=800`
- the `800 -> 1200` segment improved by `0.0630` bpb
- the gap to `1.5` is now `0.1760` bpb

This is not enough to declare success, but it is enough to keep the branch alive until the `step=1600` comparison.

That comparison has now landed:

```text
step=1600 train_loss=4.4516 train_bpb=1.5058 val_loss=4.7778 val_bpb=1.6143 muon_lr=2.069e-02 adamw_lr=3.104e-04 elapsed=2397.8s
```

The result is nuanced:

```text
original d640 schedule at step=1600: 1.6102
late-cooldown d640 at step=1600:     1.6143
```

So the restart is:

```text
0.0041 bpb worse
```

That is disappointing, but not enough to kill the branch yet.

- it is still `0.0521` bpb better than the old plain `d512` curve at the same horizon
- it is still carrying a materially higher learning rate than the original schedule at this point
- the whole reason for the restart was to improve the `1600 -> 2400` segment, not just the midpoint

So the branch stays alive until the `step=2400` comparison.

The next checkpoint is the first one that starts to justify that patience:

```text
step=2000 train_loss=4.1942 train_bpb=1.3980 val_loss=4.7537 val_bpb=1.6028 muon_lr=1.355e-02 adamw_lr=2.032e-04 elapsed=2997.2s
```

That matters because:

- the restart improved again from `1.6143 -> 1.6028`
- the gap to `1.5` is now `0.1028` bpb
- the original `d640` schedule's `step=2400` score was `1.6118`
- this restarted branch is already `0.0090` bpb better than that with `400` fewer steps

So the late-cooldown idea is finally paying off in the part of training where the original schedule flattened.

The direct late-horizon comparison has now landed:

```text
step=2400 train_loss=4.2399 train_bpb=1.3699 val_loss=4.6790 val_bpb=1.5943 muon_lr=7.844e-03 adamw_lr=1.177e-04 elapsed=3596.4s
```

This is the clean apples-to-apples result:

```text
original d640 schedule at step=2400: 1.6118
late-cooldown d640 at step=2400:     1.5943
```

So the restart is now:

```text
0.0175 bpb better
```

That is the strongest confirmation yet that the restart was correct.

- the branch is still `0.0943` bpb away from `1.5`
- so it is not close enough to declare victory
- but it is now clearly the best live width-and-schedule combination in the repo
- the run should continue to `step=3200`

The completed result is now in:

```text
step=2800 train_loss=4.0143 train_bpb=1.3240 val_loss=4.6982 val_bpb=1.5935 muon_lr=4.558e-03 adamw_lr=6.837e-05 elapsed=4195.8s
=== final_stats ===
steps=3200
seconds=4795.88
final_val_loss=4.7186
final_val_bpb=1.6017
compressed_model_size_bytes=26896298
code_size_bytes=46305
total_artifact_bytes=26942603
artifact_budget_ok=False
```

So the final picture for this branch is:

- completed exact `final_val_bpb = 1.6017072436714903`
- best sampled checkpoint on the way was better at `1.5942796520064857`
- the branch improved the best completed local exact score by `0.02213872521121504` bpb
- it still missed `1.5` by `0.10170724367149031` bpb

That means the branch was worth doing, but not worth repeating. The right follow-up is the next width bracket with the same improved schedule.

That launch is already active:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d704_gqa_softcap_cd05_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=704 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1877 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d704_gqa_softcap_cd05_s3200.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d704_gqa_softcap_cd05_s3200.log
```

Launch output:

```text
parameters=22,758,392
step=0 train_loss=10.1916 train_bpb=3.3106 val_loss=10.1290 val_bpb=3.4684 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

The first real checkpoint from the new width bracket is:

```text
step=400 train_loss=5.3727 train_bpb=1.8242 val_loss=5.5879 val_bpb=1.8962 muon_lr=3.861e-02 adamw_lr=5.792e-04 elapsed=666.8s
```

That is almost identical to the completed `d640 + cd05` branch at the same point:

```text
d640 + cd05 at step=400: 1.8955
d704 + cd05 at step=400: 1.8962
```

So the new width branch is still alive, but not yet clearly better.

The next checkpoint made the decision:

```text
step=800 train_loss=5.0553 train_bpb=1.6759 val_loss=5.2106 val_bpb=1.7553 muon_lr=3.439e-02 adamw_lr=5.159e-04 elapsed=1332.4s
```

Compared with the completed `d640 + cd05` branch:

```text
d640 + cd05 at step=800: 1.7390
d704 + cd05 at step=800: 1.7553
```

That is enough to cut `d704`.

- the branch was only tied at `step=400`
- by `step=800` it was `0.0163` bpb worse
- that is not the kind of move that closes a remaining `0.1017` gap to `1.5`

So the project did not keep climbing width blindly. It moved to the next best lever: keep the winning `d640 + cd05` recipe and give it a longer horizon.

That launch is now active:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_cd05_s4800 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=4800 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd05_s4800.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd05_s4800.log
```

Launch output:

```text
parameters=20,238,248
step=0 train_loss=10.1780 train_bpb=3.3062 val_loss=10.1269 val_bpb=3.4676 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

That `cd05 s4800` run eventually reached:

```text
step=3200 train_loss=3.8179 train_bpb=1.2391 val_loss=4.7321 val_bpb=1.6063 muon_lr=1.140e-02 adamw_lr=1.710e-04 elapsed=6250.8s
```

Compared with the completed `3200`-step branch:

```text
completed 3200-step branch final: 1.6017
cd05 s4800 at step=3200:          1.6063
```

So the longer-horizon retry was:

```text
0.0046 bpb worse
```

That is close enough to be informative, but not good enough to keep spending wall-clock on the exact same schedule.

So the project made the next schedule move rather than the next width move:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_cd10_s4800 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=4800 COOLDOWN_FRACTION=0.10 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps CHECKPOINT_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.log
```

Launch output:

```text
parameters=20,238,248
checkpoint=missing path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt
step=0 train_loss=10.1780 train_bpb=3.3062 val_loss=10.1269 val_bpb=3.4676 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
checkpoint=saved path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt step=1
```

The first real checkpoint from the new schedule branch is:

```text
step=400 train_loss=5.3829 train_bpb=1.8277 val_loss=5.5889 val_bpb=1.8966 muon_lr=3.931e-02 adamw_lr=5.897e-04 elapsed=703.7s
checkpoint=saved path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt step=401
```

Compared with `cd05 s4800` at the same point:

```text
cd05 s4800 at step=400: 1.8951
cd10 s4800 at step=400: 1.8966
```

So `cd10` is only `0.0015` bpb worse so far, which is close enough to keep alive.

The next checkpoint keeps it alive:

```text
step=800 train_loss=5.0352 train_bpb=1.6693 val_loss=5.1616 val_bpb=1.7388 muon_lr=3.716e-02 adamw_lr=5.573e-04 elapsed=1333.1s
checkpoint=saved path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt step=801
```

Compared with the best existing references:

```text
cd05 s4800 at step=800:      1.7362
completed 3200-step at 800:  1.7390
cd10 s4800 at step=800:      1.7388
```

So `cd10` is:

```text
0.0026 bpb worse than cd05 s4800
0.0002 bpb better than the completed 3200-step run
```

That is still effectively a tie, so the checkpointed `cd10` branch stays alive.

The next checkpoint keeps the same story:

```text
step=1200 train_loss=4.5467 train_bpb=1.5751 val_loss=4.9065 val_bpb=1.6769 muon_lr=3.372e-02 adamw_lr=5.057e-04 elapsed=1974.0s
checkpoint=saved path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt step=1201
```

Compared with `cd05 s4800`:

```text
cd05 s4800 at step=1200: 1.6756
cd10 s4800 at step=1200: 1.6769
```

So `cd10` is only `0.0013` bpb worse there. That is still close enough to keep alive.

## 4A. I Added A Queue Runner So Follow-Ups Can Start Immediately

While waiting on the live `cd10` branch, I added a small serial runner:

```text
queue_runs.py
```

It reads a JSON manifest of experiments and runs them one after another, writing:

- per-run logs
- per-run stats JSON
- a small leaderboard JSONL/JSON summary

The first prepared queue is:

```text
queues/d640_followups.json
```

with these next branches:

```text
bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800
bytelevel24k_d576_gqa_softcap_cd05_s4800
```

I dry-ran the queue and compiled the file successfully:

```text
[1/2] bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800
[2/2] bytelevel24k_d576_gqa_softcap_cd05_s4800
```

This is not a score improvement by itself, but it makes the next long runs cheaper to keep in motion.

## 4B. The `cd10` Schedule Retune Lost The Decision And I Cut It

The next checkpoint finally made the decision:

```text
step=1600 train_loss=4.4688 train_bpb=1.5116 val_loss=4.7834 val_bpb=1.6162 muon_lr=2.928e-02 adamw_lr=4.392e-04 elapsed=2778.8s
checkpoint=saved path=/Users/aryan/Desktop/golf/runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_cd10_s4800.pt step=1601
```

Compared with the two closest references:

```text
best completed cd05 s3200 at step=1600: 1.6143
cd05 s4800 at step=1600:               1.6169
cd10 s4800 at step=1600:               1.6162
```

So `cd10` was:

- `0.0007` bpb better than the already-weaker `cd05 s4800` retry
- `0.0019` bpb worse than the best completed `cd05 s3200` branch

That is not enough separation to justify spending more wall-clock on schedule-only tuning.

So I cut it and promoted the first queued fallback:

```text
bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800
```

The new live branch started cleanly:

```text
[1/2] run_id=bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800
train_batch_size_tokens=32768 train_microbatch_size_tokens=16384 grad_accum_steps=2
step=0 train_loss=10.1729 train_bpb=3.3852 val_loss=10.1254 val_bpb=3.4671 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

## 4C. I Also Prepared The Next Knob Queue Behind The Width Queue

While the accumulated `d640` branch is running, I encoded the next best local knob tests into:

```text
queues/d640_knob_queue.json
```

The queued branches are:

```text
bytelevel24k_d640_gqa_softcap_cd08_w100_s3200
bytelevel24k_d640_gqa_qg175_cd05_s1600
bytelevel24k_d640_gqa_softcap_nosmear_cd05_s1600
```

That queue dry-runs cleanly:

```text
[1/3] bytelevel24k_d640_gqa_softcap_cd08_w100_s3200
[2/3] bytelevel24k_d640_gqa_qg175_cd05_s1600
[3/3] bytelevel24k_d640_gqa_softcap_nosmear_cd05_s1600
```

So if the width queue still fails, the next three local levers are already encoded:

- better warmup plus midpoint cooldown
- higher query gain
- no-smear ablation

## 5. What I Think Now

The search tree is now:

1. Let the live `bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800` accumulation branch reach `step=400`.
2. If effective batch helps clearly, keep accumulation as the main line.
3. If it still stalls, let the queue continue into `bytelevel24k_d576_gqa_softcap_cd05_s4800`.
4. Keep the `relu2 + block_scales + resid_mix` branch as a prepared fallback, not the first next move.
5. Treat `48k` and `64k` as contingency denominator branches only after accumulated `24k` width stops paying.

Why width is now the lead hypothesis:

- the original `d640` run was `0.0562` bpb better than the old plain `d512` curve at matched `step=1600`
- it was still `0.0418` bpb better at `step=2400`
- that is outside the local proxy noise band we have been using
- the gain survived long enough that it no longer looks like a warmup artifact
- the only thing that broke was the late-stage improvement rate, which points to schedule before it points away from width

Why I am not prioritizing `48k` or `relu2` first:

- the healthiest `48k` local branch still lost clearly by `step=1600`
- the `relu2` family underperformed both in the all-at-once form and in the simpler core-only form
- the current width line is closer to `1.5` than either of those branches ever got at similar maturity

## 6. Reviewer Questions

These are the questions I would most want external feedback on now:

1. Is restarting `d640` with a later cooldown the right reaction to the `step=1600 -> step=2400` plateau, or should the project have finished the original run before changing schedule?
2. If this late-cooldown retry still stalls, is `d704` the right next move, or is that likely to overrun local optimization quality on this hardware?
3. Is there a better way to spend extra capacity in the `24k` family than pure width, given that tokenizer scaling beyond `24k` has not yet paid off locally?
4. Are there baseline-inspired tweaks more promising than the current `relu2 + block_scales + resid_mix` branch, which now looks second-tier?

## 7. Current Bottom Line

The best completed local exact result is still:

```text
1.6017072436714903
```

The most important stopped comparison run is:

```text
bytelevel24k_d640_gqa_softcap_cd05_s4800
step=3200 val_bpb=1.6063
```

The active frontier is:

```text
bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800
step=0 val_bpb=3.4671
```

The prepared model-side ablation is:

```text
MLP_KIND=relu2
USE_BLOCK_SCALES=1
USE_RESID_MIX=1
```

We are still not at `1.5`, but the project now has both of the remaining obvious levers active or prepared:

- larger effective token budget on the best `24k d640` recipe
- a new model-side branch beyond the current GQA/softcap recipe
