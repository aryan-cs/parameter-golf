# Parameter Golf Starter

This repo started as a standalone plan. It now includes the first working slice of that plan:

- `train_gpt.py`: a runnable looped-transformer training starter with Muon/AdamW splitting, delayed QAT hooks, artifact-size reporting, and synthetic-data smoke mode.
- `train_tokenizer.py`: a ByteLevel BPE trainer for local text corpora or optional Hugging Face dataset streaming.
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

`train_gpt.py` can also sample from local packed token files:

- `.npy` arrays containing token ids
- flat `.bin` files readable with `numpy.memmap`

Example:

```bash
DATA_PATH=./data/tokens \
AVG_BYTES_PER_TOKEN=3.5 \
MAX_STEPS=200 \
uv run python train_gpt.py
```

`AVG_BYTES_PER_TOKEN` is used to convert cross-entropy into estimated bpb. Once the tokenizer and dataset export are fixed, this should be measured from the real validation set.

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

## Recommended next steps

1. Wire in the actual FineWeb token export path and record measured bytes/token.
2. Add a `records/<submission_name>/` packaging flow once training outputs are real.
3. Add multi-GPU training and test-time training only after the single-process path is stable.
