# DRAFT1

Prepared: March 18, 2026

This file is a reviewer handoff for the entire project so far. The goal is to let another strong engineer or researcher answer four questions quickly:

1. What has already been built?
2. What exact experiments have already been tried?
3. What worked, what failed, and what is still only a proxy?
4. Are we on a good path toward a real Parameter Golf submission, or should we pivot?

This document is intentionally redundant. It overlaps with `STRATEGY.md`, `RULES.md`, `PLAN.md`, the git history, and the run artifacts. The point is to make external review easy.

## 1. One-Page Executive Summary

Project goal:

- Build a competitive OpenAI Parameter Golf submission.
- Current explicit near-term target: get local exact-tokenizer experiments down to at least `1.5` bpb before thinking seriously about a record-track PR.

Current status:

- We are not submission-ready.
- We do understand the official rules and packaging requirements.
- We have built both:
  - a local proxy-search pipeline for fast experimentation
  - an official-path alignment layer for shard ingestion, packaging, and exact tokenizer-aware `val_bpb`

Current best numbers:

- Official main-track score to beat: `1.22436570`
- Official notable non-record comparison: `1.20737944`
- Best historical local proxy result:
  - run id: `fineweb16k_d496_l4`
  - exact `final_val_bpb`: `3.1106495880562357`
  - artifact: `15,999,366` bytes
- Best completed local exact-tokenizer result:
  - run id: `torch_sp1024_d512_l4_s100_v16`
  - exact sampled `best_val_bpb`: `2.580798659621066`
  - exact sampled `final_val_bpb`: `2.7162548714625445`
  - artifact: `3,530,127` bytes
- Best live checkpoint observed so far:
  - run id: `torch_sp1024_d512_l4_b64k_s200`
  - checkpoint: `step=50`
  - exact sampled `val_bpb`: `2.4833`
  - status: run still active at the time this draft was written

Main conclusion right now:

- The old local proxy pipeline helped us discover useful architecture/tokenizer directions.
- But the fastest promising loop now is:
  - official `sp1024` data
  - official SentencePiece tokenizer semantics
  - exact sampled `val_bpb`
  - PyTorch on MPS, not MLX, for local search speed

What we still do not have:

- A competition-valid under-10-minute `8xH100` run
- A full official-validation score that is strong enough to matter
- A score anywhere near `1.5`, let alone `1.22436570`

## 2. Competition Understanding and Constraints

Primary official source:

- https://github.com/openai/parameter-golf/tree/main

Local rules summary:

- `RULES.md`

Key constraints we are working under:

- Main record track is best `val_bpb`
- Artifact cap is decimal `16,000,000` bytes
- Artifact is compressed model bytes plus code bytes
- Evaluation must not depend on network or training-data access
- Record runs must reproduce under `10` minutes on `8xH100`
- Validation is the fixed full `fineweb_val_*` split
- Tokenizer changes are allowed but heavily scrutinized
- Record submissions are PRs that add one self-contained folder under `records/...`

Important official references:

- Main leader:
  - `Naive Baseline`
  - exact `val_bpb = 1.22436570`
- Notable non-record:
  - `4-Hour Baseline`
  - exact `val_bpb = 1.20737944`

Practical implication:

- Most of our early work is useful but proxy-only.
- We should not confuse local short-run sample-slice scores with leaderboard scores.

## 3. Environment and Hardware Assumptions

Primary development environment so far:

- macOS on Apple Silicon
- local repo at `/Users/aryan/Desktop/golf`
- `uv` as the package/environment manager
- MPS used for fast PyTorch experiments
- MLX tested for official Mac-path alignment

Important local facts discovered:

- `uv sync --extra mlx` installs MLX correctly
- MPS works locally
- MLX official-path runs are correct but too slow on this Mac to be the main search loop

## 4. Current Repository Structure and What Each File Is For

Main code:

- `train_gpt.py`
  - main local trainer
  - now supports:
    - synthetic training
    - packed local token shards
    - official headered shard ingestion
    - exact SentencePiece `val_bpb`
- `train_tokenizer.py`
  - trains local or streamed tokenizers
- `prepare_tokens.py`
  - converts text into packed local train/val shards plus metadata
- `sweep.py`
  - multi-config sweep runner
- `autopilot.py`
  - width-frontier runner with persistent logs and result summaries
- `cache_text.py`
  - local raw-text cache for repeated tokenizer experiments
- `package_record.py`
  - builds official-style `records/...` folders from finished runs
- `run_official_mlx.py`
  - wrapper around the official `train_gpt_mlx.py`
  - saves logs and parses final exact metric lines

Documentation:

- `PLAN.md`
  - working strategy and theory
- `RULES.md`
  - official rule alignment summary
- `STRATEGY.md`
  - lab notebook with commands, outputs, and interpretations
- `README.md`
  - usage notes and project overview

Data and runs:

- `data/raw/`
  - cached text corpora
- `data/tokenizers/`
  - generated tokenizer artifacts
- `data/tokens/`
  - packed token shards and metadata
- `runs/`
  - structured run outputs and JSON stats

## 5. Full Git / Change History

Everything below happened on March 18, 2026.

`1344028` `initial commit`

- Repository base state before the main build-out.

`5d6f565` `Add Parameter Golf starter training scaffold`

- Created the first runnable trainer scaffold.
- Included the looped transformer idea, QAT path, Muon + AdamW split, and artifact-size reporting.

`8f4d1a6` `Switch project environment management to uv`

- Replaced ad hoc environment steps with `uv`.
- Added `pyproject.toml` and `uv.lock`.

`f4a2958` `why is it not showing up in github`

- Poorly named commit, but important.
- Added the real token-packing workflow via `prepare_tokens.py`.

`417063f` `Auto-detect packed dataset metadata in trainer`

- Let `train_gpt.py` infer `vocab_size`, `token_dtype`, and bytes/token from packed data metadata.

`0a19ad5` `Add structured run logging and local sweep runner`

- Added `sweep.py`.
- Created structured per-run JSON outputs.

`1f02532` `Document experiment history in strategy log`

- Added `STRATEGY.md` as the lab notebook.

`e9e91ea` `Log MPS sweep results and add promoted sweep preset`

- Logged first meaningful MPS sweep results.

`c7c3561` `Add local raw text caching workflow`

- Added `cache_text.py`.
- Reduced dependence on repeated streaming reads.

`eb5c38c` `Log promoted width sweep and next tokenizer plan`

- Recorded the width-scaling decision that led to the 16k tokenizer jump.

`3c9c10e` `Add autopilot frontier runner`

- Added `autopilot.py`.

`f4b0425` `Log 16k frontier experiments and current best`

- Recorded the first strong valid 16k frontier point.

`69f2c93` `Log 24k tokenizer rejection experiment`

- Recorded that 24k looked worse than 16k for this model family under the size cap.

`888ddf0` `Document official competition rules and align plan`

- Added `RULES.md`.
- Corrected `PLAN.md` to match the official repo and deadline.

`27c5cb0` `Log official competition audit in strategy`

- Recorded the official repo review and its implications.

`859c2f9` `Track official score threshold in strategy`

- Added official score tracking into `STRATEGY.md`.

`5b10c5a` `Add official shard support and record packager`

- Added official shard ingestion support to `train_gpt.py`.
- Added `package_record.py`.

`fd6e1ce` `Add official MLX run wrapper`

- Added `run_official_mlx.py`.
- Added `logs/` to `.gitignore`.

`6ed3b96` `Log official MLX search progress`

- Logged the first official MLX alignment attempts and why they were too slow.

`a1ff57b` `Add exact SentencePiece val_bpb support`

- This is one of the most important commits so far.
- Added exact tokenizer-aware byte accounting to `train_gpt.py`.

`fe51c98` `Log exact SentencePiece MPS experiments`

- Recorded the first successful exact sampled MPS runs on official tokenizer semantics.

`7719404` `Track live 64k exact-bpb checkpoint`

- Logged the best live exact sampled checkpoint so far from the longer `64k` run.

## 6. Full Experiment History

This is the experiment-by-experiment history in chronological order.

### Experiment 1. Synthetic Trainer Smoke Test

Command:

```bash
MAX_STEPS=5 uv run python train_gpt.py
```

Result:

- Passed
- `final_val_bpb = 4.3073`
- artifact under budget at `14,666,460` bytes

Why it mattered:

- Proved the starter trainer runs end to end.

### Experiment 2. Sample-Corpus Tokenizer Smoke Test

Command:

```bash
uv run python train_tokenizer.py --input-file data/sample_corpus.txt --vocab-size 512 --output-dir /tmp/golf-tokenizer --prefix smoke
```

Result:

- Passed
- tokenizer compressed size `1,725` bytes

Why it mattered:

- Proved tokenizer training and artifact-size measurement work locally.

### Experiment 3. FineWeb 8k Tokenizer Training

Command:

```bash
uv run python train_tokenizer.py --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-split train --stream --max-docs 15000 --vocab-size 8192 --output-dir ./data/tokenizers --prefix fineweb_8k_sample
```

Result:

- Passed
- tokenizer compressed size `75,330` bytes

Why it mattered:

- First real tokenizer on real FineWeb text.

### Experiment 4. FineWeb 8k Token Packing

Command:

```bash
uv run python prepare_tokens.py --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-split train --stream --train-docs 10000 --val-docs 1000 --tokenizer-prefix ./data/tokenizers/fineweb_8k_sample --output-dir ./data/tokens/fineweb_8k_sample
```

Result:

- Passed before a later HF transport issue
- train:
  - `10,000` docs
  - `5` shards
  - `avg_bytes_per_token = 3.753234512252691`
- val:
  - `1,000` docs
  - `1` shard
  - `avg_bytes_per_token = 3.7297704797873896`

Why it mattered:

- Established the first local real-data packed-shard workflow.

### Experiment 5. Real-Data MPS Training Run on Packed 8k Shards

Command:

```bash
MAX_STEPS=20 DATA_PATH=./data/tokens/fineweb_8k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_8k_sample/val uv run python train_gpt.py
```

Result:

- Passed
- `final_val_bpb = 3.3354` at step 10 during the logged path
- final artifact `4,190,769` bytes

Why it mattered:

- First proof that the trainer learns on real FineWeb-derived packed data.

### Experiment 6. Local Sweep Smoke Run on Packed 8k Shards

Command:

```bash
.venv/bin/python sweep.py --preset m4-mini --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --output /tmp/golf-sweep/results.jsonl
```

Result:

- Passed
- CPU-only by accident
- best config from the smoke:
  - `d320_l4`
  - `final_val_bpb = 3.1595`

Why it mattered:

- First useful architecture ranking.

### Experiment 7. Sweep Runner Labeling Fix Smoke Test

Command:

```bash
.venv/bin/python sweep.py --preset m4-mini --limit 1 --max-steps 1 --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --output /tmp/golf-sweep2/results.jsonl
```

Result:

- Passed
- verified `RUN_ID` was propagated correctly

Why it mattered:

- Made later sweeps auditable.

### Experiment 8. Corrected MPS Sweep on Packed 8k Shards

Command:

```bash
uv run python sweep.py --preset m4-mini --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_m4/results.jsonl
```

Result:

- Passed
- ranking:
  - `m4_d320_l4 = 3.2697`
  - `m4_d256_l4 = 3.3253`
  - `m4_d256_l6 = 3.3334`
  - `m4_d192_l4 = 3.3446`

What we learned:

- width helped
- extra loops did not help

### Experiment 9. Promoted Width Sweep on Packed 8k Shards

Command:

```bash
uv run python sweep.py --preset m4-promote --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_promote/results.jsonl
```

Result:

- Passed
- ranking:
  - `m4_d512_l4 = 3.1385`
  - `m4_d448_l4 = 3.2192`
  - `m4_d384_l4 = 3.2483`
  - `m4_d320_l4 = 3.2697`

What we learned:

- width kept helping up to `512`
- artifact still only `9,760,363` bytes
- that justified a tokenizer-size increase

### Experiment 10. Train a 16k Tokenizer and Pack 16k Local Shards

Command:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 16384 --output-dir ./data/tokenizers --prefix fineweb_16k_sample
.venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_16k_sample --output-dir ./data/tokens/fineweb_16k_sample
```

Result:

- Passed
- tokenizer compressed size `159,955` bytes
- validation `avg_bytes_per_token = 4.0827647853877345`

What we learned:

- `16k` materially improved the denominator versus `8k`

### Experiment 11. Single 16k Promotion Run with the Previous Best 8k Architecture

Command:

```bash
RUN_ID=fineweb16k_d512_l4 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 MAX_STEPS=20 DATA_PATH=./data/tokens/fineweb_16k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_16k_sample/val .venv/bin/python train_gpt.py
```

Result:

- Passed, but invalid
- `final_val_bpb = 3.0802`
- artifact `16,587,517` bytes
- over budget

What we learned:

- `16k` improved the metric
- old `d512_l4` no longer fit

### Experiment 12. 16k Width Sweep on CPU to Find the New Budget Frontier

Command:

```bash
MAX_STEPS=20 .venv/bin/python sweep.py --preset m4-promote --data-path ./data/tokens/fineweb_16k_sample/train --val-data-path ./data/tokens/fineweb_16k_sample/val --output runs/fineweb_16k_sample_promote_cpu/results.jsonl
```

Result:

- Passed
- best valid point from this sweep:
  - `d448_l4 = 2.9547`
- best invalid point:
  - `d512_l4 = 2.9013`

What we learned:

- the 16k tokenizer beat the older 8k regime
- but we were now right on the size frontier

### Experiment 13. Host-Side MPS Frontier Sweep with `autopilot.py`

Command:

```bash
MAX_STEPS=20 .venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_16k_sample/train --val-data-path ./data/tokens/fineweb_16k_sample/val --widths 448 464 480 496 512 --max-steps 20 --device mps --output-dir runs/fineweb_16k_frontier_mps --run-prefix fineweb16k
```

Result:

- Passed after one sandbox-only MPS failure
- valid ranking:
  - `fineweb16k_d496_l4 = 3.1106495880562357`
  - `fineweb16k_d448_l4 = 3.1128`
  - `fineweb16k_d480_l4 = 3.1153`
  - `fineweb16k_d464_l4 = 3.1155`
- invalid best:
  - `fineweb16k_d512_l4 = 3.0801`

Most important takeaway:

- `fineweb16k_d496_l4` became the best under-budget proxy point
- artifact `15,999,366` bytes
- only `634` bytes of headroom

### Experiment 14. 24k Tokenizer Frontier Rejection Test

Command:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 24576 --output-dir ./data/tokenizers --prefix fineweb_24k_sample
.venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_24k_sample --output-dir ./data/tokens/fineweb_24k_sample
.venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_24k_sample/train --val-data-path ./data/tokens/fineweb_24k_sample/val --widths 352 384 416 448 --max-steps 20 --output-dir runs/fineweb_24k_frontier_cpu --run-prefix fineweb24k
```

Result:

- Passed
- valid best:
  - `fineweb24k_d352_l4 = 3.2041`
- invalid best:
  - `fineweb24k_d416_l4 = 3.1012`
- tokenizer validation bytes/token improved again to `4.249964966564089`

What we learned:

- `24k` looked worse overall than `16k` for this model family
- this was an important negative result

### Experiment 15. Official Repository Audit and Rules Alignment

Command:

```bash
git clone --depth 1 https://github.com/openai/parameter-golf /tmp/parameter-golf-official.BbMmsz
```

Result:

- Passed
- confirmed:
  - exact leader `1.22436570`
  - fixed validation split
  - records-folder submission format
  - no network/data access during evaluation

What we learned:

- our local metrics were proxy metrics, not leaderboard-like scores

### Experiment 16. Official Shard Ingestion Smoke Test

Command:

```bash
DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 SEQ_LEN=128 TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 MAX_STEPS=1 DEVICE=cpu .venv/bin/python train_gpt.py
```

Result:

- Passed
- proved `train_gpt.py` could ingest official shard layout and header format

### Experiment 17. Official-Style Record Packaging Smoke Test

Command:

```bash
tmpout=$(mktemp -d /tmp/golf-records-smoke.XXXXXX) && .venv/bin/python package_record.py --stats runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.json --log runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.log --name "FineWeb 16k d496 l4" --slug fineweb16k_d496_l4 --author "Aryan" --github-id aryan-cs --blurb "Local proxy run packaged for records-folder iteration." --track-dir track_non_record_16mb --output-root "$tmpout"
```

Result:

- Passed
- generated:
  - `README.md`
  - `submission.json`
  - `train.log`
  - `train_gpt.py`

What we learned:

- we can package runs in the official folder shape

### Experiment 18. Official MLX Smoke on Real `sp1024` Data

Command:

```bash
RUN_ID=official_mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model .venv/bin/python /tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
```

Result:

- Aborted manually after step `200/200`
- training was healthy
- final evaluation was too slow because validation chunks were effectively tiny

What we learned:

- the official MLX path is correct
- but our first local evaluation geometry was bad

### Experiment 19. Expand Official `sp1024` Cache from 1 to 10 Train Shards

Command:

```bash
.venv/bin/python /tmp/parameter-golf-official.BbMmsz/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Result:

- Passed
- moved from `1` train shard to `10`

### Experiment 20. Baseline-Scale Official MLX Attempt on 10 Shards

Command:

```bash
.venv/bin/python run_official_mlx.py --official-root /tmp/parameter-golf-official.BbMmsz --run-id official_mlx_s10_i600_tb524k_mb32k --output-dir runs/official_mlx_sp1024 --data-path /tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 --tokenizer-path /tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model --env ITERATIONS=600 --env TRAIN_BATCH_TOKENS=524288 --env VAL_BATCH_SIZE=524288 --env TRAIN_LOG_EVERY=50 --env MAX_WALLCLOCK_SECONDS=0 --env MLX_MAX_MICROBATCH_TOKENS=32768
```

Result:

- Failed with `SIGKILL`

What we learned:

- that shape was too aggressive for the local machine

### Experiment 21. Safer Official MLX Medium-Batch Run on 10 Shards

Command:

```bash
.venv/bin/python run_official_mlx.py --official-root /tmp/parameter-golf-official.BbMmsz --run-id official_mlx_s10_i1200_tb262k_mb16k_logit8k --output-dir runs/official_mlx_sp1024 --data-path /tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 --tokenizer-path /tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model --env ITERATIONS=1200 --env TRAIN_BATCH_TOKENS=262144 --env VAL_BATCH_SIZE=262144 --env TRAIN_LOG_EVERY=100 --env MAX_WALLCLOCK_SECONDS=0 --env MLX_MAX_MICROBATCH_TOKENS=16384 --env LOGIT_CHUNK_TOKENS=8192
```

Follow-up relaunch:

```bash
.venv/bin/python run_official_mlx.py --official-root /tmp/parameter-golf-official.BbMmsz --run-id official_mlx_s10_i1000_tb262k_mb16k_logit8k_w0 --output-dir runs/official_mlx_sp1024 --data-path /tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 --tokenizer-path /tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model --env ITERATIONS=1000 --env TRAIN_BATCH_TOKENS=262144 --env VAL_BATCH_SIZE=262144 --env TRAIN_LOG_EVERY=50 --env MAX_WALLCLOCK_SECONDS=0 --env MLX_MAX_MICROBATCH_TOKENS=16384 --env LOGIT_CHUNK_TOKENS=8192 --env WARMUP_STEPS=0
```

Result:

- memory-stable
- too slow
- sample signal:
  - `step:1/1000 ... tok_s:3453`

What we learned:

- MLX official-path is too slow on this Mac for the main local search loop

### Experiment 22. Tiny Official-Shard Smoke for Exact SentencePiece `val_bpb`

Command:

```bash
TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH="$tmpdir" VAL_DATA_PATH="$tmpdir" VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=64 N_HEADS=4 D_FF=170 N_LOOPS=2 SEQ_LEN=128 TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 VAL_STEPS=0 VAL_LOSS_EVERY=0 MAX_STEPS=1 DEVICE=cpu .venv/bin/python train_gpt.py
```

Result:

- Passed
- `bpb_mode=sentencepiece_exact`
- `final_val_bpb = 4.0297`

What we learned:

- exact SentencePiece byte accounting works in our fast trainer

### Experiment 23. First Exact Sampled-Validation MPS Run on Official `sp1024`

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_s100_v16 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=100 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_s100_v16.json .venv/bin/python train_gpt.py
```

Result:

- Passed
- best sampled checkpoint:
  - `best_val_bpb = 2.580798659621066`
- final:
  - `final_val_bpb = 2.7162548714625445`

What we learned:

- this was the first really useful exact-tokenizer local run
- schedule needed improvement

### Experiment 24. `64k` Batch Throughput Probe on the Exact Sampled MPS Path

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s20 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=32768 VAL_STEPS=4 VAL_LOSS_EVERY=10 MAX_STEPS=20 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=0.98 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s20.json .venv/bin/python train_gpt.py
```

Result:

- Passed
- `final_val_bpb = 3.4061132475273808`
- runtime `132.00s`

What we learned:

- `64k` batch is viable
- tokens per second improved enough to justify longer runs

### Experiment 25. Longer `64k` Batch Exact Sampled Run

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s200 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=200 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=0.98 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s200.json .venv/bin/python train_gpt.py
```

Live output observed so far:

```text
step=0 train_loss=7.0376 train_bpb=4.1401 val_loss=7.0164 val_bpb=4.1588 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=50 train_loss=4.2241 train_bpb=2.4879 val_loss=4.1846 val_bpb=2.4833 muon_lr=1.865e-02 adamw_lr=2.798e-04 elapsed=156.9s
```

Runtime status at time of writing:

```text
32232 54:55  15.7  0.5 Us   .venv/bin/python train_gpt.py
```

What we learned so far:

- This is the best exact sampled checkpoint seen so far.
- The longer schedule seems better than the earlier `100`-step run.

## 7. What Has Worked

Things that look directionally correct:

- Using `16k` instead of `8k` was a good move for the older proxy loop.
- Width helped more than extra loops in the early local sweeps.
- Spending time on infrastructure was worthwhile:
  - metadata auto-detection
  - structured run logging
  - raw-text caching
  - official shard support
  - record packaging
  - exact tokenizer-aware `val_bpb`
- The fast PyTorch/MPS trainer is currently a better local iteration engine than the official MLX path.
- Exact SentencePiece accounting in the fast trainer was probably the most important recent improvement.
- Larger train batches on MPS look viable and worthwhile.

## 8. What Has Not Worked

Clear dead ends or partial dead ends:

- Treating early local proxy results as if they were close to official scores
- `24k` tokenizer for the current architecture family
- Using the official MLX path as the main local search loop
- Relying on live Hugging Face streaming during every experiment
- Early short-run schedules with aggressive cooldown and early QAT

## 9. Main Risks / Places We Might Be Wrong

These are the biggest review targets.

Risk 1:

- We may still be optimizing too much for sampled validation behavior and not enough for full official validation.

Risk 2:

- The looped-transformer family we started with may simply be too weak relative to the official baseline family.

Risk 3:

- Our local exact sampled scores may not correlate tightly enough with full fixed-split exact scores.

Risk 4:

- We may be underusing the official `sp1024` family and should be closer to the baseline architecture/schedule before inventing too much.

Risk 5:

- We may be over-investing in local Mac optimization when the real score frontier will only become visible on H100.

## 10. Questions I Want a Reviewer to Answer

This is the most important section if someone is reviewing our direction.

Question 1:

- Is the current pivot correct?
- Specifically: was it the right call to move from the slow official MLX path to the faster MPS trainer with exact SentencePiece byte accounting?

Question 2:

- Is `sp1024` now the right main local regime, or should we still keep spending time on custom tokenizers?

Question 3:

- Is the looped/shared-block architecture still worth pursuing, or should we move closer to the official baseline family first and only then experiment?

Question 4:

- Should the next real improvement attempt come from:
  - larger train batch
  - longer run
  - schedule tuning
  - architecture change
  - or moving sooner to cloud GPU experiments

Question 5:

- Is sampled exact validation a good enough local selection metric, or should we build a cheaper but still more faithful partial-full-val evaluator?

Question 6:

- Are there obvious architecture changes missing from our current code that are much more likely to matter than the current width/schedule tuning?

## 11. What I Think the Reviewer Will Probably Say

My guess:

- The official-path alignment work was necessary and good.
- The early proxy-only tokenizer frontier work was useful but not enough by itself.
- The current exact SentencePiece MPS path is the first local loop that feels genuinely useful.
- We still need a stronger architecture or much better schedule to get remotely near `1.5`.
- We should probably avoid spending much more time on 24k or bigger tokenizers right now.

## 12. Immediate Next Actions If No Reviewer Stops Us

If nobody interrupts this direction, the next steps are:

1. Let the current `torch_sp1024_d512_l4_b64k_s200` run finish and record the final sampled exact score.
2. Compare that final score to the live `2.4833` checkpoint.
3. If the longer run keeps helping, continue:
   - larger batch or longer horizon
   - later QAT
   - less aggressive cooldown
4. Once a clearly better candidate exists, run a full exact validation scan on that checkpoint instead of doing it every iteration.
5. Only after that, decide whether to:
   - widen further
   - change architecture family
   - or move to H100 for the next search stage

## 13. Bottom Line

We are not close to submission quality yet, but we are no longer wandering blindly.

The project has moved through three distinct phases:

1. Build a working local scaffold.
2. Explore tokenizer and width frontiers with cheap proxy loops.
3. Align with the official competition path and build a faster exact-tokenizer local search loop.

The third phase is where the project finally started to feel technically grounded.

The best evidence for that is:

- official rules are understood
- official shard ingestion works
- official packaging works
- exact SentencePiece `val_bpb` works in the fast trainer
- the best live exact sampled checkpoint is now `2.4833`

That is still far from `1.5`, and very far from `1.22436570`, but it is the first point where the search loop looks believable enough to keep investing in.
