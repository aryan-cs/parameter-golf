# Parameter Golf Starter

This repo started as a standalone plan. It now includes the first working slice of that plan:

- `train_gpt.py`: a runnable looped-transformer training starter with Muon/AdamW splitting, delayed QAT hooks, artifact-size reporting, and synthetic-data smoke mode.
- `train_tokenizer.py`: a ByteLevel BPE trainer for local text corpora or optional Hugging Face dataset streaming.
- `cache_text.py`: a raw-text caching utility that snapshots a reusable local train/val corpus for tokenizer experiments.
- `prepare_tokens.py`: a token-packing utility that writes local `train/` and `val/` token shards plus bytes/token metadata.
- `pyproject.toml`: project metadata and `uv`-managed dependencies.

The current implementation is meant to get us from ideas to executable code quickly. It focuses on the first high-leverage pieces from `PLAN.md`:

- 32k tokenizer workflow
- looped transformer architecture
- Muon optimizer integration
- quantization-aware training hooks
- compressed artifact measurement

What is not built yet:

- distributed multi-GPU training
- official FineWeb export/re-tokenization pipeline
- test-time training evaluation
- submission record packaging under `records/`

## Install

```bash
uv sync
```

If you want to experiment with the MLX path on Apple Silicon, install the optional extra:

```bash
uv sync --extra mlx
```

## Smoke test

This runs on synthetic tokens so you can verify the training loop without downloading data.

```bash
MAX_STEPS=5 \
VOCAB_SIZE=4096 \
D_MODEL=128 \
N_HEADS=4 \
N_LOOPS=2 \
SEQ_LEN=64 \
TRAIN_BATCH_TOKENS=2048 \
uv run python train_gpt.py
```

## Local token data

`train_gpt.py` can sample from local packed token files:

- `.npy` arrays containing token ids
- flat `.bin` files readable with `numpy.memmap`

If the token directory contains a `metadata.json`, the trainer now auto-detects `vocab_size`, `token_dtype`, and `avg_bytes_per_token`.

Example using pre-packed local tokens:

```bash
DATA_PATH=./data/tokens \
MAX_STEPS=200 \
uv run python train_gpt.py
```

Override `AVG_BYTES_PER_TOKEN` only if you want to force a custom value.

## Tokenizer training

Train from local text files:

```bash
uv run python train_tokenizer.py \
  --input-dir ./data/raw_text \
  --glob '*.txt' \
  --vocab-size 32768 \
  --output-dir ./data/tokenizers \
  --prefix fineweb_32k_bpe
```

You can also train from cached `.jsonl` text files produced by `cache_text.py`.

Optional Hugging Face streaming mode:

```bash
uv run python train_tokenizer.py \
  --hf-dataset HuggingFaceFW/fineweb \
  --hf-config sample-10BT \
  --hf-split train \
  --stream \
  --max-docs 50000 \
  --vocab-size 32768 \
  --output-dir ./data/tokenizers \
  --prefix fineweb_32k_bpe
```

## Packing Real Token Shards

For repeatable tokenizer experiments, first cache raw text locally:

```bash
uv run python cache_text.py \
  --hf-dataset HuggingFaceFW/fineweb \
  --hf-config sample-10BT \
  --hf-split train \
  --stream \
  --train-docs 15000 \
  --val-docs 1000 \
  --output-dir ./data/raw/fineweb_sample
```

Then train a tokenizer from the cached training corpus:

```bash
uv run python train_tokenizer.py \
  --input-file ./data/raw/fineweb_sample/train.jsonl \
  --vocab-size 16384 \
  --output-dir ./data/tokenizers \
  --prefix fineweb_16k_sample
```

After training a tokenizer, pack local `train/` and `val/` shards plus metadata:

```bash
uv run python prepare_tokens.py \
  --train-input-file ./data/raw/fineweb_sample/train.jsonl \
  --val-input-file ./data/raw/fineweb_sample/val.jsonl \
  --tokenizer-prefix ./data/tokenizers/fineweb_16k_sample \
  --output-dir ./data/tokens/fineweb_16k_sample
```

Then train against the packed shards:

```bash
DATA_PATH=./data/tokens/fineweb_16k_sample/train \
VAL_DATA_PATH=./data/tokens/fineweb_16k_sample/val \
MAX_STEPS=200 \
uv run python train_gpt.py
```

Each training run now writes a structured stats JSON file and prints its path at the end. Set `STATS_PATH` if you want to choose the filename explicitly.

## Local Sweep

Run a small local ablation sweep over a few model sizes:

```bash
uv run python sweep.py \
  --preset m4-mini \
  --data-path ./data/tokens/fineweb_32k_sample/train \
  --val-data-path ./data/tokens/fineweb_32k_sample/val \
  --max-steps 20
```

This writes per-run JSON summaries under `runs/` and a JSONL summary file you can compare across experiments.

For a quick sanity check of just the first config, add `--limit 1`.

Current presets:

- `m4-mini`: small local architecture sweep
- `m4-promote`: width-focused follow-up sweep after picking a local winner

## Autopilot Frontier Runs

For repeated budget-frontier checks, use `autopilot.py`. It keeps a per-run `.log`, writes structured JSON summaries, and prints both overall and under-budget rankings.

Example:

```bash
uv run python autopilot.py \
  --data-path ./data/tokens/fineweb_16k_sample/train \
  --val-data-path ./data/tokens/fineweb_16k_sample/val \
  --widths 448 464 480 496 512 \
  --max-steps 20 \
  --device mps \
  --output-dir runs/fineweb_16k_frontier_mps \
  --run-prefix fineweb16k
```

Add `--resume` to skip runs that already have a saved summary.

## Record Packaging

Once a run is interesting enough to preserve, package it into an official-style `records/...` folder:

```bash
uv run python package_record.py \
  --stats runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.json \
  --log runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.log \
  --name "FineWeb 16k d496 l4" \
  --slug fineweb16k_d496_l4 \
  --author "Your Name" \
  --github-id your-github-id \
  --blurb "Local proxy run packaged for record-folder iteration." \
  --track-dir track_non_record_16mb
```

This generates:

- `records/<track>/<date>_<slug>/README.md`
- `records/<track>/<date>_<slug>/submission.json`
- `records/<track>/<date>_<slug>/train.log`
- `records/<track>/<date>_<slug>/train_gpt.py`

Use this only for runs that are actually worth preserving. A generated folder is not automatically competition-valid; it still needs the official evaluation path and the right track semantics.

## Recommended next steps

1. Scale the `prepare_tokens.py` workflow from a small sample to larger FineWeb-derived runs.
2. Use `sweep.py` to find the best local shape before spending time on larger tokenizers or longer runs.
3. Add a `records/<submission_name>/` packaging flow once training outputs are real.
4. Add multi-GPU training and test-time training only after the single-process path is stable.
