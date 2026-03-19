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
- The larger-tokenizer path is alive, but currently optimization-limited.
- We still have not reached `1.5`.

Current best numbers worth knowing:

- Official main-track score to beat: `1.22436570`
- Practical clean-record target after the `0.005 nats` significance bar:
  - `1.217152224795555` bpb
- Best overall sampled local exact result so far:
  - `torch_sp1024_d512_l4_b64k_s200`
  - `best_val_bpb = 1.995522745133331`
- Best long-context sampled local exact result so far:
  - `torch_sp1024_d512_l4_b32k_s400_seq1024_cd20`
  - `best_val_bpb = 2.0050325515775427`
- Best exact `24k` tokenizer result so far:
  - `bytelevel24k_d512_tied_s400`
  - `final_val_bpb = 2.0819050756023527`

Main conclusion right now:

- The search no longer has an infrastructure bottleneck.
- The current bottleneck is getting the promising exact `24k` branch enough effective batch and enough optimization quality to keep dropping below the low `2`s.

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

### 2.4 The `24k` tokenizer was re-opened and is no longer a dead branch

Before the feedback, the `24k` tokenizer had been treated as basically rejected.
That is no longer justified.

What actually happened after re-testing:

- `24k` exact, tied embeddings, `40` steps:
  - `2.593778352406914`
- `24k` exact, tied embeddings, `400` steps:
  - `2.0819050756023527`
- `16k` exact, tied embeddings, `40` steps:
  - `2.6767`

So:

- `24k` is clearly better than `16k` in this regime.
- It is not yet good enough.
- The current problem seems to be optimization / effective batch, not a broken tokenizer path.

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

- `bytelevel24k_d512_tied_s400`
  - `final_val_bpb = 2.0819050756023527`

Supporting results:

- `bytelevel24k_d512_tied_s40`
  - `2.593778352406914`
- `bytelevel24k_d512_tied_l8_s100`
  - `2.3681701384086318`

Interpretation:

- deeper loops did not rescue the `24k` branch
- the branch improves steadily but flattens too high
- this suggests we need more effective batch and/or a stronger optimizer/training setup, not just more depth

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
- the next honest step is effective `64k` through accumulation, not another architecture pivot

## 5. Current Best Path Toward `1.5`

Right now, the best path does not look like "go back to the old proxy width sweep" or "copy the official baseline harder."

It looks like:

1. Stay in the exact ByteLevel `24k` branch.
2. Use tied embeddings.
3. Use `SEQ_LEN=1024`.
4. Use accumulation to increase effective train batch beyond the direct-memory limit.
5. Only after that, revisit width or optimizer choices if the score still stalls.

Why this seems best:

- `24k` clearly beat `16k`
- exact byte accounting is now real
- the branch already reaches `2.0819`
- the next missing ingredient is clearly effective batch / optimization, not missing infrastructure

## 6. What Still Looks Wrong or Incomplete

### 6.1 We still do not have a path to `1.5`

That is the blunt truth.

The best overall sampled exact score is still `1.9955`.
The best long-context score is `2.0050`.
The best exact `24k` score is `2.0819`.

We are meaningfully better than where we were at `DRAFT1`, but still far from the local target.

### 6.2 The larger-tokenizer branch is still over the artifact cap

Even with tied embeddings:

- `24k`, `d_model=512` is still far over
- `16k`, `d_model=512` is only slightly over, but also scores worse

So if the `24k` branch eventually becomes the best score path, it will still need a later artifact-compression / smaller-width / different-model pass.

### 6.3 We have not yet tested the strongest next accumulated-batch run

The most obvious next run is:

```bash
PYTHONUNBUFFERED=1 \
RUN_ID=bytelevel24k_eff64k_accum_s400 \
TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample \
DATA_PATH=./data/tokens/fineweb_24k_sample/train \
VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val \
D_MODEL=512 \
N_HEADS=8 \
D_FF=1365 \
N_LOOPS=4 \
SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_MICROBATCH_TOKENS=16384 \
VAL_BATCH_TOKENS=16384 \
VAL_STEPS=16 \
VAL_LOSS_EVERY=100 \
MAX_STEPS=400 \
COOLDOWN_FRACTION=0.20 \
QAT_START_FRACTION=1.0 \
TIED_EMBEDDINGS=1 \
DEVICE=mps \
uv run python train_gpt.py
```

That run has not been completed yet at the time of this draft.

## 7. Reviewer Questions

If another strong reviewer picks this up, the most useful questions now are:

1. Is the shared-loop family still the right one once we are in exact `24k` territory, or are we wasting time not changing the optimizer/model more aggressively?
2. Is `24k` actually the correct tokenizer target, or should we be exploring something closer to `32k` with a smaller model width?
3. Is there an obvious optimizer/schedule change that should replace the current Muon + AdamW split for the large-vocab branch?
4. Should we add grouped-query attention next, or is accumulated batch the more important lever?
5. If the real goal is a leaderboard submission rather than a local `1.5`, should we stop chasing large-vocab local scores and move earlier to an artifact-aware branch?

## 8. Bottom Line

Compared with `DRAFT1`, the project is in a much better state:

- feedback was actually integrated, not just acknowledged
- the long-context regime is validated
- the exact ByteLevel tokenizer path exists
- tied embeddings exist
- gradient accumulation exists

But the bottom-line score has still not crossed `1.5`.

The best current interpretation is:

- the search is no longer confused
- the next step is now obvious
- the remaining problem is still hard

That is better than where we were, but it is not victory yet.
