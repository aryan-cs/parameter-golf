# Strategy Log

This file is the running lab notebook for the project.

Rules for updating this file:

1. Every meaningful code change should be recorded here.
2. Every experiment should include the exact command that was run.
3. Every experiment should include actual terminal output, not just a paraphrase.
4. Each entry should say what we learned and what to try next.
5. If a run fails, log the failure and the exact error.

## Goal

Build toward a strong Parameter Golf submission with:

- a tokenizer that improves bytes/token materially
- a compact looped transformer that fits well under the 16MB artifact cap
- a reproducible local workflow for tokenizer training, token packing, and ablation sweeps

## Current Best Logged Real-Data Result

Dataset:

- FineWeb `sample-10BT` streamed from Hugging Face
- Local tokenizer: `fineweb_8k_sample`
- Packed local shards:
  - `data/tokens/fineweb_8k_sample/train`
  - `data/tokens/fineweb_8k_sample/val`

Best real-data terminal result currently logged:

- `val_bpb=3.3354` at step 10 on MPS during a 20-step run with:
  - `d_model=256`
  - `n_heads=8`
  - `d_ff=682`
  - `n_loops=4`
  - `vocab_size=8192`

Most recent final value from that 20-step MPS run:

- final compressed model size: `4,163,585` bytes
- total artifact size: `4,190,769` bytes

## Code Strategy Timeline

### 1. Starter Training Scaffold

Commit:

- `5d6f565` `Add Parameter Golf starter training scaffold`

Purpose:

- Build a runnable looped-transformer starter before wiring real data.

Key code idea:

```python
class LoopedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_loops, max_seq_len, use_smear):
        super().__init__()
        self.n_loops = n_loops
        self.embed = nn.Embedding(vocab_size, d_model)
        self.smear = SmearBlock(d_model) if use_smear else nn.Identity()
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        self.block = SharedTransformerBlock(d_model, n_heads, d_ff)
        self.loop_embeddings = nn.Embedding(n_loops, d_model)
        self.final_norm = RMSNorm(d_model)
        self.head = QATLinear(d_model, vocab_size, bias=False)
```

Why this matters:

- The project needed a minimal but credible architecture skeleton before data work or ablations would mean anything.

### 2. Switch Environment Management to `uv`

Commit:

- `8f4d1a6` `Switch project environment management to uv`

Purpose:

- Make dependency setup reproducible and fast.

Key code/config:

```toml
[project]
name = "parameter-golf-starter"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.18",
  "numpy>=1.26",
  "sentencepiece>=0.2",
  "tokenizers>=0.15",
  "torch>=2.3",
  "tqdm>=4.66",
]
```

Why this matters:

- We needed a stable, repeatable environment before pushing on tokenizer training or packed-data runs.

### 3. Add Real Token Packing Workflow

Commit:

- `f4a2958` `why is it not showing up in github`

Note:

- The commit message is not descriptive, but this commit is important because it introduced `prepare_tokens.py`.

Purpose:

- Bridge the gap between raw text and the trainer by generating local packed `train/` and `val/` token shards.

Key code:

```python
for index, text in enumerate(itertools.islice(documents, total_needed)):
    token_ids = tokenizer.encode(text)
    byte_count = len(text.encode("utf-8"))
    if index < args.val_docs:
        val_writer.add_document(token_ids, byte_count)
    else:
        train_writer.add_document(token_ids, byte_count)
```

Why this matters:

- Before this, the project could train only on synthetic tokens or hand-prepared arrays.
- After this, we could stream FineWeb text, tokenize it, and produce local shards with measured `avg_bytes_per_token`.

### 4. Auto-Detect Packed Dataset Metadata in the Trainer

Commit:

- `417063f` `Auto-detect packed dataset metadata in trainer`

Purpose:

- Remove manual `VOCAB_SIZE`, `TOKEN_DTYPE`, and `AVG_BYTES_PER_TOKEN` plumbing for packed datasets.

Key code:

```python
cfg.vocab_size, vocab_size_source = resolve_vocab_size(cfg, val_path)
cfg.token_dtype, token_dtype_source = resolve_token_dtype(cfg, val_path)
avg_bytes_per_token, avg_bytes_source = resolve_avg_bytes_per_token(cfg, val_path)
```

Why this matters:

- This reduced command-line friction and removed a common source of run-configuration mistakes.

### 5. Add Structured Run Logging and Local Sweep Runner

Commit:

- `0a19ad5` `Add structured run logging and local sweep runner`

Purpose:

- Turn one-off terminal runs into comparable experiments with saved metrics and rankings.

Key code from `train_gpt.py`:

```python
summary = {
    "run_id": cfg.run_id,
    "device": str(device),
    "train_source": train_source,
    "val_source": val_source,
    "vocab_size": cfg.vocab_size,
    "token_dtype": cfg.token_dtype,
    "avg_bytes_per_token": avg_bytes_per_token,
    "parameters": param_count,
    "steps": step,
    "seconds": total_time,
    "best_val_bpb": best_val_bpb,
    "final_val_bpb": final_val_bpb,
    "compressed_model_size_bytes": len(compressed),
    "total_artifact_bytes": total_artifact,
}
```

Key code from `sweep.py`:

```python
PRESETS = {
    "m4-mini": [
        {"run_id": "m4_d192_l4", "D_MODEL": "192", "N_HEADS": "6", "D_FF": "512", "N_LOOPS": "4"},
        {"run_id": "m4_d256_l4", "D_MODEL": "256", "N_HEADS": "8", "D_FF": "682", "N_LOOPS": "4"},
        {"run_id": "m4_d256_l6", "D_MODEL": "256", "N_HEADS": "8", "D_FF": "682", "N_LOOPS": "6"},
        {"run_id": "m4_d320_l4", "D_MODEL": "320", "N_HEADS": "8", "D_FF": "853", "N_LOOPS": "4"},
    ]
}
```

Why this matters:

- We can now compare local ablations systematically instead of relying on memory and scrollback.

## Experiment Log

## Experiment 1. Synthetic Trainer Smoke Test

Status:

- Passed

Purpose:

- Verify the core trainer runs end-to-end before touching real data.

Command:

```bash
MAX_STEPS=5 uv run python train_gpt.py
```

Terminal output:

```text
config: {'run_id': 'dev_smoke', 'data_path': '', 'val_data_path': '', 'token_dtype': 'uint16', 'vocab_size': 32768, 'd_model': 256, 'n_heads': 8, 'd_ff': 682, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 5, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': 3.5, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': ''}
train_source=synthetic val_source=synthetic device=mps
parameters=17,565,248
step=0 train_loss=10.4475 train_bpb=4.3065 val_loss=10.4496 val_bpb=4.3073 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=3 qat=enabled
=== final_stats ===
steps=5
seconds=1.95
compressed_model_size_bytes=14641858
code_size_bytes=24602
total_artifact_bytes=14666460
artifact_budget_ok=True
```

Interpretation:

- The starter model trains and quantizes end-to-end.
- The artifact budget check already works.
- QAT enablement triggers correctly.

Next step:

- Replace synthetic data with real tokenizer and packed local shards.

## Experiment 2. Sample-Corpus Tokenizer Smoke Test

Status:

- Passed

Purpose:

- Verify tokenizer training and size measurement locally.

Command:

```bash
uv run python train_tokenizer.py --input-file data/sample_corpus.txt --vocab-size 512 --output-dir /tmp/golf-tokenizer --prefix smoke
```

Terminal output:

```text
[00:00:00] Pre-processing sequences       ██████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ██████████████████████████ 91       /       91
[00:00:00] Count pairs                    ██████████████████████████ 91       /       91
[00:00:00] Compute merges                 ██████████████████████████ 95       /       95
source=local-files
vocab_path=/tmp/golf-tokenizer/smoke-vocab.json
merges_path=/tmp/golf-tokenizer/smoke-merges.txt
raw_size_bytes=3704
compressed_size_bytes=1725
```

Interpretation:

- Tokenizer training is functioning.
- The compressed tokenizer size reporting is functioning.

Next step:

- Train a real tokenizer on a FineWeb sample instead of the toy corpus.

## Experiment 3. FineWeb 8k Tokenizer Training

Status:

- Passed

Purpose:

- Create the first real tokenizer backed by FineWeb sample text.

Command:

```bash
uv run python train_tokenizer.py --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-split train --stream --max-docs 15000 --vocab-size 8192 --output-dir ./data/tokenizers --prefix fineweb_8k_sample
```

Terminal output:

```text
README.md: 44.3kB [00:00, 30.4MB/s]       ██████████████████████████ 0        /        0
Resolving data files: 100%|███████████████████| 27468/27468 [00:00<00:00, 115202.23it/s]
[00:00:10] Pre-processing sequences       ██████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ██████████████████████████ 241157   /   241157
[00:00:00] Count pairs                    ██████████████████████████ 241157   /   241157
[00:00:00] Compute merges                 ██████████████████████████ 7935     /     7935
source=hf:HuggingFaceFW/fineweb
vocab_path=data/tokenizers/fineweb_8k_sample-vocab.json
merges_path=data/tokenizers/fineweb_8k_sample-merges.txt
raw_size_bytes=178493
compressed_size_bytes=75330
```

Interpretation:

- First real tokenizer artifact is small: `75,330` compressed bytes.
- 8k vocab is very manageable within the challenge budget.

Next step:

- Pack local train and val shards from the same tokenizer.

## Experiment 4. FineWeb 8k Token Packing

Status:

- Passed

Purpose:

- Turn streamed FineWeb text into local packed token arrays with metadata.

Command:

```bash
uv run python prepare_tokens.py --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-split train --stream --train-docs 10000 --val-docs 1000 --tokenizer-prefix ./data/tokenizers/fineweb_8k_sample --output-dir ./data/tokens/fineweb_8k_sample
```

Terminal output:

```text
Resolving data files: 100%|███████████████████| 27468/27468 [00:00<00:00, 113665.34it/s]
{
  "source": "hf:HuggingFaceFW/fineweb",
  "output_dir": "data/tokens/fineweb_8k_sample",
  "tokenizer_prefix": "./data/tokenizers/fineweb_8k_sample",
  "train": {
    "split": "train",
    "source": "hf:HuggingFaceFW/fineweb",
    "tokenizer_prefix": "./data/tokenizers/fineweb_8k_sample",
    "vocab_size": 8192,
    "token_dtype": "uint16",
    "docs": 10000,
    "shards": 5,
    "total_bytes": 31031514,
    "total_tokens": 8267939,
    "avg_bytes_per_token": 3.753234512252691
  },
  "val": {
    "split": "val",
    "source": "hf:HuggingFaceFW/fineweb",
    "tokenizer_prefix": "./data/tokenizers/fineweb_8k_sample",
    "vocab_size": 8192,
    "token_dtype": "uint16",
    "docs": 1000,
    "shards": 1,
    "total_bytes": 3032792,
    "total_tokens": 813131,
    "avg_bytes_per_token": 3.7297704797873896
  }
}
```

Additional terminal output from the chained run:

```text
'[Errno 9] Bad file descriptor' thrown while requesting GET https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/9bb295ddab0e05d785b879661af7260fed5140fc/sample/10BT/000_00000.parquet
Retrying in 1s [Retry 1/5].
^Cerror: Failed to get PID of child process
  Caused by: ESRCH: No such process
```

Interpretation:

- The important part succeeded before the Hugging Face transport error.
- We have usable local shards on disk:
  - `train`: 10,000 docs, 5 shards
  - `val`: 1,000 docs, 1 shard
- The actual measured validation bytes/token is about `3.7298`, which is much better than the synthetic default.

Next step:

- Train the model purely from local packed shards to avoid depending on live HF streaming during iteration.

## Experiment 5. Real-Data MPS Training Run on Packed 8k Shards

Status:

- Passed

Purpose:

- Confirm the trainer learns on packed FineWeb-derived data and uses metadata automatically.

Command:

```bash
MAX_STEPS=20 DATA_PATH=./data/tokens/fineweb_8k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_8k_sample/val uv run python train_gpt.py
```

Terminal output:

```text
config: {'run_id': 'dev_smoke', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 256, 'n_heads': 8, 'd_ff': 682, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': ''}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=4,982,336
step=0 train_loss=9.0753 train_bpb=3.5104 val_loss=9.0642 val_bpb=3.5061 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.6763 train_bpb=3.3560 val_loss=8.6231 val_bpb=3.3354 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=1.8s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=3.77
compressed_model_size_bytes=4163585
code_size_bytes=27184
total_artifact_bytes=4190769
artifact_budget_ok=True
```

Interpretation:

- This is the first clear proof that the model is training on real packed FineWeb-derived data.
- The loss and bpb both improved within 20 steps.
- The artifact is still far under budget, leaving room for larger models and larger tokenizers.

What this suggests:

- We are currently under-using the artifact budget.
- Scaling model width and/or loop count is likely worthwhile.

Next step:

- Compare a few local model sizes systematically instead of manually.

## Experiment 6. Local Sweep Smoke Run on Packed 8k Shards

Status:

- Passed

Purpose:

- Compare several local model shapes using the same packed dataset.

Command:

```bash
.venv/bin/python sweep.py --preset m4-mini --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --output /tmp/golf-sweep/results.jsonl
```

Important note:

- This smoke sweep ran on CPU because it was invoked via `.venv/bin/python` without an explicit `--device mps`.
- The run still produced a useful relative ranking for quick local ablation direction.

Terminal output summary:

```text
=== sweep_ranking ===
1. dev_smoke final_val_bpb=3.1595 params=6474000 artifact=5428523
2. dev_smoke final_val_bpb=3.2209 params=4982336 artifact=4202629
3. dev_smoke final_val_bpb=3.2402 params=3589696 artifact=3032965
4. dev_smoke final_val_bpb=3.2422 params=4982848 artifact=4207060
results_path=/tmp/golf-sweep/results.jsonl
```

Detailed per-config outputs from that sweep:

```text
m4_d192_l4:
final_val_bpb=3.2402
params=3,589,696
artifact=3,032,965

m4_d256_l4:
final_val_bpb=3.2209
params=4,982,336
artifact=4,202,629

m4_d256_l6:
final_val_bpb=3.2422
params=4,982,848
artifact=4,207,060

m4_d320_l4:
final_val_bpb=3.1595
params=6,474,000
artifact=5,428,523
```

Interpretation:

- Among the tested shapes, `d_model=320, n_loops=4` was best.
- Increasing width from 256 to 320 helped more than increasing loops from 4 to 6 at the same width.
- At this scale, we are still far below the 16MB artifact budget.

Fix made after this run:

- `sweep.py` now passes `RUN_ID` correctly and supports `--limit` for quick checks.

## Experiment 7. Sweep Runner Labeling Fix Smoke Test

Status:

- Passed

Purpose:

- Confirm the sweep runner writes the correct `RUN_ID` into saved stats and ranking output.

Command:

```bash
.venv/bin/python sweep.py --preset m4-mini --limit 1 --max-steps 1 --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --output /tmp/golf-sweep2/results.jsonl
```

Terminal output:

```text
[1/1] run_id=m4_d192_l4
config: {'run_id': 'm4_d192_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 192, 'n_heads': 6, 'd_ff': 512, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 1, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': '/tmp/golf-sweep2/m4_d192_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=cpu
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=3,589,696
step=0 train_loss=9.0137 train_bpb=3.4865 val_loss=8.8951 val_bpb=3.4407 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=1.15
final_val_loss=8.8949
final_val_bpb=3.4406
compressed_model_size_bytes=3022679
code_size_bytes=29425
total_artifact_bytes=3052104
artifact_budget_ok=True
stats_path=/tmp/golf-sweep2/m4_d192_l4.json

=== sweep_ranking ===
1. m4_d192_l4 final_val_bpb=3.4406 params=3589696 artifact=3052104
results_path=/tmp/golf-sweep2/results.jsonl
```

Interpretation:

- The `RUN_ID` bug is fixed.
- The sweep runner is ready for actual MPS sweeps.

## Current Working Hypothesis

The next most likely win on the current local setup is:

1. Run the corrected MPS sweep over the 8k tokenizer sample.
2. Promote the best local architecture from that sweep.
3. Increase tokenizer size from 8k toward 16k or 32k once the architecture baseline is stable.

Reasoning:

- We are still significantly below the artifact size cap.
- The 8k tokenizer already gives around `3.73` bytes/token on validation, which is promising.
- Local ablations suggest width is helping more than extra loop count at this stage.

## Next Command To Run

This is the next command we want to execute:

```bash
uv run python sweep.py --preset m4-mini --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_m4/results.jsonl
```

Why this exact command:

- It uses the corrected sweep runner.
- It uses the already prepared local packed FineWeb shards.
- It evaluates multiple model shapes on the real MPS path rather than CPU smoke mode.
