# Parameter Golf Starter

This repo started as a standalone plan. It now includes the first working slice of that plan:

- `train_gpt.py`: a runnable looped-transformer training starter with Muon/AdamW splitting, delayed QAT hooks, artifact-size reporting, and synthetic-data smoke mode.
- `train_tokenizer.py`: a ByteLevel BPE trainer for local text corpora or optional Hugging Face dataset streaming.
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

After training a tokenizer, pack local `train/` and `val/` shards plus metadata:

```bash
uv run python prepare_tokens.py \
  --hf-dataset HuggingFaceFW/fineweb \
  --hf-config sample-10BT \
  --hf-split train \
  --stream \
  --train-docs 10000 \
  --val-docs 1000 \
  --tokenizer-prefix ./data/tokenizers/fineweb_32k_bpe \
  --output-dir ./data/tokens/fineweb_32k_sample
```

Then train against the packed shards:

```bash
DATA_PATH=./data/tokens/fineweb_32k_sample/train \
VAL_DATA_PATH=./data/tokens/fineweb_32k_sample/val \
MAX_STEPS=200 \
uv run python train_gpt.py
```

## Recommended next steps

1. Scale the `prepare_tokens.py` workflow from a small sample to larger FineWeb-derived runs.
2. Add a `records/<submission_name>/` packaging flow once training outputs are real.
3. Add multi-GPU training and test-time training only after the single-process path is stable.
