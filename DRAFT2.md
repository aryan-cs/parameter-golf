# DRAFT2

Prepared: March 18, 2026

This is the current reviewer packet after the work recorded in `DRAFT1.md`.

`DRAFT1.md` captured the project up through the first exact `sp1024` local runs and the initial external review cycle.
This file captures what happened after that review and what it changed.

The goal of this draft is to let a reviewer answer five questions quickly:

1. Did we actually follow the feedback?
2. What changed in the code?
3. What exact new experiments were run?
4. What is the new best path toward `1.5`?
5. What still looks wrong or incomplete?

## 1. Executive Update

The short version:

- We did follow the feedback.
- Fixing context length was the correct direction.
- The best long-context local result is now within `0.0095` bpb of the old sampled best.
- Exact ByteLevel BPE experiments are now real, not proxy-only.
- The larger-tokenizer path is now the best overall exact local branch, but it is still optimization- and artifact-limited.
- We still have not reached `1.5`.

Current best numbers worth knowing:

- Official main-track score to beat: `1.22436570`
- Practical clean-record target after the `0.005 nats` significance bar:
  - `1.217152224795555` bpb
- Best overall local exact result so far:
  - `bytelevel24k_d512_tied_lr2x_s1600`
  - `final_val_bpb = 1.7853542005916354`
- Best official-tokenizer sampled local exact result so far:
  - `torch_sp1024_d512_l4_b64k_s200`
  - `best_val_bpb = 1.995522745133331`
- Best long-context sampled local exact result so far:
  - `torch_sp1024_d512_l4_b32k_s400_seq1024_cd20`
  - `best_val_bpb = 2.0050325515775427`
- Best exact `24k` tokenizer result so far:
  - `bytelevel24k_d512_tied_lr2x_s1600`
  - `final_val_bpb = 1.7853542005916354`

Main conclusion right now:

- The search no longer has an infrastructure bottleneck.
- The current bottleneck is pushing the promising exact `24k` branch the rest of the way to `1.5` while eventually pulling its artifact size back down.

## 2. What Changed Since `DRAFT1`

### 2.1 Rules and score tracking got corrected

We added and logged the unit conversion that matters for submission decisions:

```text
0.005 nats = 0.007213475204444817 bpb
1.22436570 - 0.007213475204444817 = 1.217152224795555
```

This means a serious leaderboard claim should target about `1.21715` bpb or better, not merely anything slightly below `1.22436570`.

### 2.2 The long-context criticism was tested directly

The external reviewer said the biggest mistake was `SEQ_LEN=128`.
We followed that.

Key result:

- `SEQ_LEN=1024`, `32k` batch, `400` steps, `COOLDOWN_FRACTION=0.20`
- `best_val_bpb = 2.0050325515775427`

This is not yet a big enough score breakthrough, but it means the long-context branch is no longer speculative.
It is now essentially tied with the old best sampled result.

### 2.3 Exact ByteLevel BPE support now exists

This is the largest code-side capability upgrade after `DRAFT1`.

`train_gpt.py` now supports:

- exact SentencePiece `val_bpb`
- exact ByteLevel BPE `val_bpb`
- `TIED_EMBEDDINGS=1`
- `SHARE_BLOCKS=0` for a baseline-like untied stack
- `TRAIN_MICROBATCH_TOKENS` for gradient accumulation

That means our `16k` and `24k` tokenizer experiments are no longer based only on metadata averages.

### 2.4 The `24k` tokenizer was re-opened and is now the main local scoring branch

Before the feedback, the `24k` tokenizer had been treated as basically rejected.
That is no longer justified.

What actually happened after re-testing:

- `24k` exact, tied embeddings, `40` steps:
  - `2.593778352406914`
- `24k` exact, tied embeddings, `400` steps:
  - `2.0819050756023527`
- `24k` exact, tied embeddings, `400` steps, `2x` LR:
  - `2.0389181133431182`
- `24k` exact, tied embeddings, `800` steps, `2x` LR:
  - `1.9059446644848441`
- `24k` exact, tied embeddings, `1600` steps, `2x` LR:
  - `1.7853542005916354`
- `16k` exact, tied embeddings, `40` steps:
  - `2.6767`

So:

- `24k` is clearly better than `16k` in this regime.
- the `24k` branch is now better than every local official-tokenizer run too
- it is still not good enough
- the current problem seems to be optimization horizon and artifact size, not a broken tokenizer path

## 3. New Code Capabilities

### 3.1 Exact ByteLevel BPE byte accounting

We added a new helper in `train_gpt.py`:

```python
class ByteLevelBPBHelper:
    def __init__(self, byte_length_lut) -> None:
        self.byte_length_lut = byte_length_lut
```

The important consequence is that tokenized `16k` / `24k` experiments can now compute exact target-byte counts directly from the ByteLevel vocabulary instead of relying only on average bytes per token from dataset metadata.

### 3.2 Tied embeddings

We added:

```python
if self.tied_embeddings:
    self.head.linear.weight = self.embed.weight
```

Why it matters:

- large vocabularies were paying a huge parameter penalty for an untied output head
- tied embeddings are closer to the official baseline family
- the larger-tokenizer path was not worth testing seriously without this

### 3.3 Untied stack option

We added:

```python
if self.share_blocks:
    ...
else:
    for block in self.blocks:
        x = block(x, cos, sin)
```

This gave us a way to test a more official-looking `9`-layer untied stack without deleting the shared-loop model family.

### 3.4 Gradient accumulation

We added:

```python
train_microbatch_tokens = cfg.train_microbatch_tokens or cfg.train_batch_tokens
train_micro_bsz = max(1, train_microbatch_tokens // cfg.seq_len)
grad_accum_steps = max(1, math.ceil(train_bsz / train_micro_bsz))
```

and then accumulate gradients across microbatches before stepping the optimizers.

Why it matters:

- the `24k` branch could smoke-test at `32k` train tokens but OOM during sustained training
- accumulation is now the clean way to test larger effective batches without changing the search direction again

### 3.5 Baseline-aligned GQA and logit softcap knobs

We also added three official-baseline-aligned controls to `train_gpt.py` behind env flags:

- `NUM_KV_HEADS`
- `QK_GAIN_INIT`
- `LOGIT_SOFTCAP`

Representative code:

```python
self.q_gain = nn.Parameter(torch.full((n_heads,), qk_gain_init))
...
if self.kv_repeat > 1:
    k = k.repeat_interleave(self.kv_repeat, dim=1)
    v = v.repeat_interleave(self.kv_repeat, dim=1)
...
if self.logit_softcap > 0.0:
    logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
```

Why it matters:

- the official baseline uses grouped-query attention
- it also uses per-head query gain and output logit softcapping
- this gives us a clean next architecture ablation if the plain `24k` branch starts flattening

## 4. Key Experiment Results Since `DRAFT1`

### 4.1 Long-context official `sp1024` branch

These are the high-signal results:

- `torch_sp1024_d512_l4_b32k_s400_seq1024`
  - `final_val_bpb = 2.0147442651306453`
- `torch_sp1024_d512_l4_b32k_s400_seq1024_cd30`
  - `final_val_bpb = 2.0257938571260774`
- `torch_sp1024_d512_l4_b32k_s400_seq1024_cd20`
  - `final_val_bpb = 2.0050325515775427`

Interpretation:

- `COOLDOWN_FRACTION=0.20` is the best of the tested local cooldowns
- the long-context branch is effectively tied with the old best sampled result
- pure schedule tuning is now yielding only small gains

### 4.2 Exact `24k` tokenizer branch

Best exact result so far:

- `bytelevel24k_d512_tied_lr2x_s1600`
  - `final_val_bpb = 1.7853542005916354`

Supporting results:

- `bytelevel24k_d512_tied_s40`
  - `2.593778352406914`
- `bytelevel24k_d512_tied_lr2x_s40`
  - `2.486486775516235`
- `bytelevel24k_d512_tied_lr3x_s40`
  - `2.4673788844876348`
- `bytelevel24k_d512_tied_lr2x_s400`
  - `2.0389181133431182`
- `bytelevel24k_d512_tied_lr2x_s800`
  - `1.9059446644848441`
- `bytelevel24k_d512_tied_l8_s100`
  - `2.3681701384086318`
- `bytelevel24k_eff64k_accum_s100`
  - `2.3006986857952625`
- `bytelevel24k_eff64k_accum_lr2x_s100`
  - `2.2311532389896986`

Interpretation:

- deeper loops did not rescue the `24k` branch
- raw bigger batch by itself did not rescue it either
- the strongest new lever inside this branch has been learning-rate retuning plus more training horizon
- this branch is now the best overall local exact line we have
- the branch still improves steadily, but it remains far over the artifact cap
- doubling the run horizon from `800` to `1600` improved the best exact local score by `0.12059046389320871` bpb

### 4.3 Exact `16k` tokenizer branch

- `bytelevel16k_d512_tied_s40`
  - `2.6767`

Interpretation:

- clearly weaker than `24k` under the same approximate regime
- does not look like the right branch right now

### 4.4 Baseline-family untied stack

- `sp1024_stack9_s100`
  - `final_val_bpb = 2.5413`

Interpretation:

- useful comparison
- not competitive enough in the current local regime
- does not currently beat the shared-loop family

### 4.5 Accumulated-batch `24k` branch

Smoke:

- `bytelevel24k_eff32k_accum_smoke`
  - accumulation works

Pilot:

- `bytelevel24k_eff32k_accum_s40`
  - `final_val_bpb = 2.5815`

Interpretation:

- this is only a small improvement over the direct `16k`-batch `24k` run
- but it proves the code path and removes the hardware blocker
- the larger effective batch alone did not solve the branch
- the more important discovery was that learning-rate retuning helps this branch more than accumulation by itself

## 5. Current Best Path Toward `1.5`

Right now, the best path does not look like "go back to the old proxy width sweep" or "copy the official baseline harder."

It looks like:

1. Stay in the exact ByteLevel `24k` branch.
2. Use tied embeddings.
3. Use `SEQ_LEN=1024`.
4. Use the promoted `2x` learning rate.
5. Use the longer `1600`-step horizon as the new default schedule.
6. Test official-baseline attention choices on top of that strong schedule before touching width again.
7. Use accumulation only when the direct-memory limit blocks a genuinely better regime.
8. Do not let artifact-size blindness hide the fact that this branch will eventually need a smaller valid variant.

Why this seems best:

- `24k` clearly beat `16k`
- exact byte accounting is now real
- the branch now reaches `1.7854`
- width and deeper loops have not moved it much
- optimization horizon is still the most responsive lever
- the next clean question is whether GQA plus logit softcap helps on the same long-horizon schedule

## 6. What Still Looks Wrong or Incomplete

### 6.1 We still do not have a complete path to `1.5`

That is still the blunt truth.

The best overall local exact score is now `1.7854`.
The best official-tokenizer sampled exact score is `1.9955`.
The best long-context official-tokenizer score is `2.0050`.

So we are materially better than where we were at `DRAFT1`, but still `0.2854` bpb away from the local `1.5` target.

### 6.2 The larger-tokenizer branch is still badly over the artifact cap

Even with tied embeddings:

- `24k`, `d_model=512`, `1600` steps produced `22,405,986` artifact bytes
- the cap is `16,000,000`, so this branch is over by `6,405,986` bytes
- `16k`, `d_model=512` is smaller, but it also scores much worse locally

So if the `24k` branch eventually becomes the best score path, it will still need a later artifact-compression / smaller-width / different-model pass.

### 6.3 We have not yet tested the strongest next baseline-aligned continuation

The most obvious next run is now:

```bash
PYTHONUNBUFFERED=1 \
RUN_ID=bytelevel24k_d512_gqa_softcap_s1600 \
TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample \
DATA_PATH=./data/tokens/fineweb_24k_sample/train \
VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val \
D_MODEL=512 \
N_HEADS=8 \
NUM_KV_HEADS=4 \
D_FF=1365 \
N_LOOPS=4 \
SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=16384 \
VAL_BATCH_TOKENS=16384 \
VAL_STEPS=16 \
VAL_LOSS_EVERY=400 \
MAX_STEPS=1600 \
COOLDOWN_FRACTION=0.20 \
QAT_START_FRACTION=1.0 \
TIED_EMBEDDINGS=1 \
MUON_LR=0.04 \
ADAMW_LR=0.0006 \
QK_GAIN_INIT=1.5 \
LOGIT_SOFTCAP=30 \
DEVICE=mps \
uv run python train_gpt.py
```

That is the cleanest next continuation because the plain `1600`-step run is now our best local result, and we now have a ready-to-test set of official-baseline attention knobs that can be layered on top without changing the rest of the regime.

## 7. Reviewer Questions

If another strong reviewer picks this up, the most useful questions now are:

1. Is the shared-loop family still the right one once we are in exact `24k` territory, or are we wasting time not changing the optimizer/model more aggressively?
2. Is `24k` actually the correct tokenizer target, or should we be exploring something closer to `32k` with a smaller model width?
3. Is there an obvious optimizer/schedule change that should replace the current Muon + AdamW split for the large-vocab branch?
4. Does the new baseline-aligned attention path suggest we should copy more of the official inductive biases before another pure optimization extension?
5. If the real goal is a leaderboard submission rather than a local `1.5`, should we stop chasing large-vocab local scores and move earlier to an artifact-aware branch?

## 8. Bottom Line

Compared with `DRAFT1`, the project is in a much better state:

- feedback was actually integrated, not just acknowledged
- the long-context regime is validated
- the exact ByteLevel tokenizer path exists
- tied embeddings exist
- gradient accumulation exists
- baseline-aligned GQA, per-head query gain, and logit softcap knobs exist
- the `24k` branch now has a real exact local score in the `1.7`s

But the bottom-line score has still not crossed `1.5`.

The best current interpretation is:

- the search is no longer confused
- the next step is now obvious
- the remaining problem is still hard

That is better than where we were, but it is not victory yet.
