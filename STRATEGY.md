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

## Official Leaderboard Reference

As of March 18, 2026, the official repo leaderboard shows:

- rank 1: `Naive Baseline`
- displayed README score: `1.2244`
- exact record score from `submission.json`: `1.22436570`
- record folder: `records/track_10min_16mb/2026-03-17_NaiveBaseline`
- exact printed post-quant roundtrip metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`

Official notable non-record reference:

- run: `4-Hour Baseline`
- displayed README score: `1.2074`
- exact record score from `submission.json`: `1.20737944`
- track: non-record unlimited compute under the same `16,000,000` byte cap

## Scoreboard Snapshot

This section should always answer two questions:

1. What exact official score do we have to beat?
2. Does our current best result actually qualify for a PR?

Current score to beat for the main leaderboard:

- exact `val_bpb` to beat: `1.22436570`
- source: official `Naive Baseline` record on March 18, 2026

Current interesting non-record comparison:

- exact `val_bpb`: `1.20737944`
- source: official `4-Hour Baseline` non-record run on March 18, 2026

Our current best local result:

- exact `final_val_bpb`: `3.1106495880562357`
- short form: `3.1106`
- run id: `fineweb16k_d496_l4`
- artifact: `15,999,366` bytes
- status: valid local proxy, not competition-valid yet

Gap versus official main-track leader:

- `3.1106495880562357 - 1.22436570 = 1.8862838880562356` bpb worse

PR opening rule:

- Open a main-track PR only when we have an exact competition-valid score strictly below `1.22436570`
- And the run must also satisfy all official submission conditions:
  - under `16,000,000` bytes total artifact size
  - reproducible under `10 minutes` on `8xH100`
  - official-style validation on the fixed full `fineweb_val_*` split
  - record folder packaging under `records/track_10min_16mb/`
  - enough evidence for the official significance and reproducibility requirements

Non-record PR rule:

- Even without beating `1.22436570`, we can still consider a PR for `records/track_non_record_16mb/` if the run is interesting, under the artifact cap, and well documented

## Current Best Logged Real-Data Result

Dataset:

- FineWeb `sample-10BT` streamed from Hugging Face
- Local raw cache:
  - `data/raw/fineweb_16k_sample/train.jsonl`
  - `data/raw/fineweb_16k_sample/val.jsonl`
- Local tokenizer: `fineweb_16k_sample`
- Packed local shards:
  - `data/tokens/fineweb_16k_sample/train`
  - `data/tokens/fineweb_16k_sample/val`

Best real-data terminal result currently logged:

- `final_val_bpb=3.1106` on MPS from the 16k width-frontier sweep with:
  - `d_model=496`
  - `n_heads=8`
  - `d_ff=1322`
  - `n_loops=4`
  - `vocab_size=16384`

Current best artifact details:

- compressed model size: `15,969,941` bytes
- total artifact size: `15,999,366` bytes
- parameters: `19,208,220`
- budget headroom: `634` bytes

Important scope note:

- This is our best local proxy result, not a competition-valid leaderboard score.
- It does not yet use the official distributed `8xH100` evaluation path or the official `records/...` submission packaging flow.

Best currently observed but invalid comparison point:

- `final_val_bpb=3.0801` on MPS from `d_model=512`, `n_loops=4`, `vocab_size=16384`
- total artifact size: `16,587,520` bytes
- conclusion: better metric, but over the 16,000,000 byte cap

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

### 6. Add Local Raw-Text Caching Workflow

Commit:

- `c7c3561` `Add local raw text caching workflow`

Purpose:

- Stop relying on repeated live Hugging Face streaming for every tokenizer experiment.

Key code from `cache_text.py`:

```python
cached_docs = list(itertools.islice(documents, total_needed))
val_docs = cached_docs[: args.val_docs]
train_docs = cached_docs[args.val_docs : args.val_docs + args.train_docs]

val_count, val_bytes = write_jsonl(val_path, val_docs)
train_count, train_bytes = write_jsonl(train_path, train_docs)
```

Key code from `prepare_tokens.py`:

```python
if args.train_input_file or args.val_input_file:
    for text in iter_text_files([Path(path) for path in args.val_input_file]):
        token_ids = tokenizer.encode(text)
        val_writer.add_document(token_ids, len(text.encode("utf-8")))
    for text in iter_text_files([Path(path) for path in args.train_input_file]):
        token_ids = tokenizer.encode(text)
        train_writer.add_document(token_ids, len(text.encode("utf-8")))
```

Why this matters:

- Future tokenizer comparisons can now reuse a stable local raw corpus.
- This reduces Hugging Face transport flakiness and keeps tokenizer experiments more comparable.

### 7. Add Autopilot Frontier Runner

Commit:

- `3c9c10e` `Add autopilot frontier runner`

Purpose:

- Automate repeatable width-frontier searches while preserving exact per-run console logs.

Key code from `autopilot.py`:

```python
def stream_process(command: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
```

Key ranking logic:

```python
under_budget = [summary for summary in summaries if summary["artifact_budget_ok"]]
print_ranking("autopilot_under_budget_ranking", under_budget)
write_results(output_dir, summaries)
```

Why this matters:

- We now keep durable `.log` files for every training run instead of relying on terminal scrollback.
- We can search the exact budget frontier without manually rebuilding commands each time.
- This makes `STRATEGY.md` updates easier because the raw outputs now live on disk.

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

## Experiment 8. Corrected MPS Sweep on Packed 8k Shards

Status:

- Passed

Purpose:

- Compare multiple architecture shapes on the real MPS path using the same packed 8k-tokenizer dataset.

Command:

```bash
uv run python sweep.py --preset m4-mini --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_m4/results.jsonl
```

Terminal output:

```text
[1/4] run_id=m4_d192_l4
config: {'run_id': 'm4_d192_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 192, 'n_heads': 6, 'd_ff': 512, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_m4/m4_d192_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=3,589,696
step=0 train_loss=9.0137 train_bpb=3.4865 val_loss=9.0104 val_bpb=3.4853 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.7261 train_bpb=3.3753 val_loss=8.6952 val_bpb=3.3633 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=1.6s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=3.00
final_val_loss=8.6468
final_val_bpb=3.3446
compressed_model_size_bytes=3004124
code_size_bytes=29425
total_artifact_bytes=3033549
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_m4/m4_d192_l4.json
[2/4] run_id=m4_d256_l4
config: {'run_id': 'm4_d256_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 256, 'n_heads': 8, 'd_ff': 682, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_m4/m4_d256_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=4,982,336
step=0 train_loss=9.0753 train_bpb=3.5104 val_loss=9.0642 val_bpb=3.5061 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.6763 train_bpb=3.3560 val_loss=8.6231 val_bpb=3.3354 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=2.0s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=3.83
final_val_loss=8.5967
final_val_bpb=3.3253
compressed_model_size_bytes=4163605
code_size_bytes=29425
total_artifact_bytes=4193030
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_m4/m4_d256_l4.json
[3/4] run_id=m4_d256_l6
config: {'run_id': 'm4_d256_l6', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 256, 'n_heads': 8, 'd_ff': 682, 'n_loops': 6, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_m4/m4_d256_l6.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=4,982,848
step=0 train_loss=9.0388 train_bpb=3.4963 val_loss=9.0355 val_bpb=3.4950 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.6721 train_bpb=3.3544 val_loss=8.6194 val_bpb=3.3340 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=2.6s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=5.20
final_val_loss=8.6178
final_val_bpb=3.3334
compressed_model_size_bytes=4175527
code_size_bytes=29425
total_artifact_bytes=4204952
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_m4/m4_d256_l6.json
[4/4] run_id=m4_d320_l4
config: {'run_id': 'm4_d320_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 320, 'n_heads': 8, 'd_ff': 853, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_m4/m4_d320_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=6,474,000
step=0 train_loss=9.1175 train_bpb=3.5267 val_loss=9.0940 val_bpb=3.5176 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.6097 train_bpb=3.3303 val_loss=8.5457 val_bpb=3.3055 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=3.0s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=5.52
final_val_loss=8.4531
final_val_bpb=3.2697
compressed_model_size_bytes=5391543
code_size_bytes=29425
total_artifact_bytes=5420968
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_m4/m4_d320_l4.json

=== sweep_ranking ===
1. m4_d320_l4 final_val_bpb=3.2697 params=6474000 artifact=5420968
2. m4_d256_l4 final_val_bpb=3.3253 params=4982336 artifact=4193030
3. m4_d256_l6 final_val_bpb=3.3334 params=4982848 artifact=4204952
4. m4_d192_l4 final_val_bpb=3.3446 params=3589696 artifact=3033549
results_path=runs/fineweb_8k_sample_m4/results.jsonl
```

Interpretation:

- `m4_d320_l4` clearly won this sweep.
- Wider helped again; extra loops did not beat the best width-matched baseline.
- The winning run is still using only about `5.42MB` of the artifact budget, which means we have room both for larger models and for larger tokenizers.

What this suggests:

- Keep `n_loops=4` for now.
- Prefer spending extra capacity on width before spending it on more loop repetitions.
- Architecture is good enough to justify a tokenizer-size follow-up experiment now.

## Current Working Hypothesis

The next most likely win on the current local setup is:

1. Keep the current promoted architecture family width-focused.
2. Probe whether wider models on the same 8k tokenizer continue to improve.
3. If width keeps helping, then move to a larger tokenizer such as 16k while preserving the strongest local architecture.

Reasoning:

- The winning 8k run is still far below the artifact size cap.
- Width is outperforming additional loops in the local ablations we have.
- Before paying the cost of another full tokenizer prep cycle, we should see whether the current tokenizer still has easy architecture gains left.

## Next Command To Run

This is the next command we want to execute:

```bash
uv run python sweep.py --preset m4-promote --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_promote/results.jsonl
```

Why this exact command:

- It keeps the dataset fixed, which makes the architecture comparison cleaner.
- It focuses on wider models because width has been the best lever so far.
- It uses the already prepared local packed FineWeb shards and avoids another HF streaming round for now.

## Experiment 9. Promoted Width Sweep on Packed 8k Shards

Status:

- Passed

Purpose:

- Test how much additional width we can exploit on the same 8k-tokenizer dataset before switching tokenizer size.

Command:

```bash
uv run python sweep.py --preset m4-promote --data-path ./data/tokens/fineweb_8k_sample/train --val-data-path ./data/tokens/fineweb_8k_sample/val --max-steps 20 --device mps --output runs/fineweb_8k_sample_promote/results.jsonl
```

Terminal output:

```text
[1/4] run_id=m4_d320_l4
config: {'run_id': 'm4_d320_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 320, 'n_heads': 8, 'd_ff': 853, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_promote/m4_d320_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=6,474,000
step=0 train_loss=9.1175 train_bpb=3.5267 val_loss=9.0940 val_bpb=3.5176 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.6097 train_bpb=3.3303 val_loss=8.5457 val_bpb=3.3055 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=2.5s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=4.90
final_val_loss=8.4531
final_val_bpb=3.2697
compressed_model_size_bytes=5391534
code_size_bytes=29425
total_artifact_bytes=5420959
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_promote/m4_d320_l4.json
[2/4] run_id=m4_d384_l4
config: {'run_id': 'm4_d384_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 384, 'n_heads': 8, 'd_ff': 1024, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_promote/m4_d384_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=8,064,096
step=0 train_loss=9.1062 train_bpb=3.5223 val_loss=9.0903 val_bpb=3.5162 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.5415 train_bpb=3.3039 val_loss=8.4767 val_bpb=3.2788 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=3.2s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=6.16
final_val_loss=8.3979
final_val_bpb=3.2483
compressed_model_size_bytes=6751297
code_size_bytes=29425
total_artifact_bytes=6780722
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_promote/m4_d384_l4.json
[3/4] run_id=m4_d448_l4
config: {'run_id': 'm4_d448_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 448, 'n_heads': 8, 'd_ff': 1194, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_promote/m4_d448_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=9,751,280
step=0 train_loss=9.1051 train_bpb=3.5219 val_loss=9.0962 val_bpb=3.5184 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.4657 train_bpb=3.2746 val_loss=8.3767 val_bpb=3.2401 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=3.9s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=7.44
final_val_loss=8.3225
final_val_bpb=3.2192
compressed_model_size_bytes=8213480
code_size_bytes=29425
total_artifact_bytes=8242905
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_promote/m4_d448_l4.json
[4/4] run_id=m4_d512_l4
config: {'run_id': 'm4_d512_l4', 'data_path': './data/tokens/fineweb_8k_sample/train', 'val_data_path': './data/tokens/fineweb_8k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 512, 'n_heads': 8, 'd_ff': 1365, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_8k_sample_promote/m4_d512_l4.json'}
train_source=5 shard(s) val_source=1 shard(s) device=mps
vocab_size=8192 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
avg_bytes_per_token=3.7298 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_8k_sample/val
parameters=11,538,048
step=0 train_loss=9.0804 train_bpb=3.5123 val_loss=9.0604 val_bpb=3.5046 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.3256 train_bpb=3.2204 val_loss=8.2467 val_bpb=3.1899 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=4.4s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=8.50
final_val_loss=8.1138
final_val_bpb=3.1385
compressed_model_size_bytes=9730938
code_size_bytes=29425
total_artifact_bytes=9760363
artifact_budget_ok=True
stats_path=runs/fineweb_8k_sample_promote/m4_d512_l4.json

=== sweep_ranking ===
1. m4_d512_l4 final_val_bpb=3.1385 params=11538048 artifact=9760363
2. m4_d448_l4 final_val_bpb=3.2192 params=9751280 artifact=8242905
3. m4_d384_l4 final_val_bpb=3.2483 params=8064096 artifact=6780722
4. m4_d320_l4 final_val_bpb=3.2697 params=6474000 artifact=5420959
results_path=runs/fineweb_8k_sample_promote/results.jsonl
```

Interpretation:

- Width kept helping all the way up through `d_model=512`.
- The best current architecture is now `m4_d512_l4`.
- Even this stronger model is still under the 16MB artifact cap at about `9.76MB`.

What this suggests:

- The next comparison should shift from architecture to tokenizer size.
- A 16k tokenizer is the cleanest next test because it should improve bytes/token without making the artifact obviously impossible.

## Current Working Hypothesis

The next most likely win on the current local setup is:

1. Keep `d_model=512`, `n_loops=4` as the promoted architecture.
2. Move from the 8k tokenizer to a 16k tokenizer using the same cached raw text corpus.
3. Re-run the 20-step local training sanity check and compare the new bytes/token and final validation bpb.

Reasoning:

- Width is still helping strongly.
- The artifact budget still has headroom.
- The next likely gain should come from the bpb denominator via tokenizer size rather than another local architecture-only sweep.

## Next Command To Run

This is the next command we want to execute:

```bash
uv run python cache_text.py --hf-dataset HuggingFaceFW/fineweb --hf-config sample-10BT --hf-split train --stream --train-docs 15000 --val-docs 1000 --output-dir ./data/raw/fineweb_16k_sample && uv run python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 16384 --output-dir ./data/tokenizers --prefix fineweb_16k_sample && uv run python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_16k_sample --output-dir ./data/tokens/fineweb_16k_sample && RUN_ID=fineweb16k_d512_l4 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 MAX_STEPS=20 DATA_PATH=./data/tokens/fineweb_16k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_16k_sample/val uv run python train_gpt.py
```

Why this exact command:

- It uses the new local raw-text cache instead of repeating multiple live HF reads.
- It keeps the winning architecture fixed so the tokenizer comparison is cleaner.
- It tests the next most likely source of improvement directly: a larger tokenizer.

## Experiment 10. Train a 16k Tokenizer and Pack 16k Local Shards

Status:

- Passed

Purpose:

- Test whether moving from the 8k tokenizer to a 16k tokenizer improves bytes/token enough to justify the larger embedding/output tables.

Command:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 16384 --output-dir ./data/tokenizers --prefix fineweb_16k_sample && .venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_16k_sample --output-dir ./data/tokens/fineweb_16k_sample
```

Terminal output:

```text
source=local-files
vocab_path=data/tokenizers/fineweb_16k_sample-vocab.json
merges_path=data/tokenizers/fineweb_16k_sample-merges.txt
raw_size_bytes=385068
compressed_size_bytes=159955
{
  "source": "local-files",
  "output_dir": "data/tokens/fineweb_16k_sample",
  "tokenizer_prefix": "./data/tokenizers/fineweb_16k_sample",
  "train": {
    "split": "train",
    "source": "local-files",
    "tokenizer_prefix": "./data/tokenizers/fineweb_16k_sample",
    "vocab_size": 16384,
    "token_dtype": "uint16",
    "docs": 15000,
    "shards": 6,
    "total_bytes": 46410649,
    "total_tokens": 11238367,
    "avg_bytes_per_token": 4.129661275521613
  },
  "val": {
    "split": "val",
    "source": "local-files",
    "tokenizer_prefix": "./data/tokenizers/fineweb_16k_sample",
    "vocab_size": 16384,
    "token_dtype": "uint16",
    "docs": 1000,
    "shards": 1,
    "total_bytes": 3032792,
    "total_tokens": 742828,
    "avg_bytes_per_token": 4.0827647853877345
  }
}
```

Interpretation:

- The validation bytes/token denominator improved from about `3.7298` on the 8k tokenizer to `4.0828` on the 16k tokenizer.
- This is a meaningful tokenizer win and justified running a fresh architecture frontier.

## Experiment 11. Single 16k Promotion Run with the Previous Best 8k Architecture

Status:

- Passed, but over budget

Purpose:

- Keep the best 8k architecture fixed and isolate the effect of the 16k tokenizer.

Command:

```bash
RUN_ID=fineweb16k_d512_l4 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 MAX_STEPS=20 DATA_PATH=./data/tokens/fineweb_16k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_16k_sample/val .venv/bin/python train_gpt.py
```

Terminal output:

```text
config: {'run_id': 'fineweb16k_d512_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 512, 'n_heads': 8, 'd_ff': 1365, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': ''}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=19,926,656
step=0 train_loss=9.7808 train_bpb=3.4562 val_loss=9.7531 val_bpb=3.4464 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.9222 train_bpb=3.1528 val_loss=8.8297 val_bpb=3.1201 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=10.9s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=22.79
final_val_loss=8.7167
final_val_bpb=3.0802
compressed_model_size_bytes=16558092
code_size_bytes=29425
total_artifact_bytes=16587517
artifact_budget_ok=False
```

Interpretation:

- The 16k tokenizer improved the proxy metric materially relative to the 8k setup.
- The exact old best shape, `d512_l4`, no longer fits under the artifact cap.
- That made a budget-frontier sweep the obvious next move.

## Experiment 12. 16k Width Sweep on CPU to Find the New Budget Frontier

Status:

- Passed

Purpose:

- Find which width still fits under the 16MB cap once the tokenizer grows to 16k.

Command:

```bash
MAX_STEPS=20 .venv/bin/python sweep.py --preset m4-promote --data-path ./data/tokens/fineweb_16k_sample/train --val-data-path ./data/tokens/fineweb_16k_sample/val --output runs/fineweb_16k_sample_promote_cpu/results.jsonl
```

Important note:

- `MAX_STEPS=20` was set in the shell, but `sweep.py` uses its own `--max-steps` argument and therefore ran the preset at its default `30` steps.
- This was an accidental mismatch, but the results were still useful.

Terminal output:

```text
[4/4] run_id=m4_d512_l4
config: {'run_id': 'm4_d512_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 512, 'n_heads': 8, 'd_ff': 1365, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 30, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_sample_promote_cpu/m4_d512_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=19,926,656
step=0 train_loss=9.7808 train_bpb=3.4562 val_loss=9.7531 val_bpb=3.4464 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.9222 train_bpb=3.1528 val_loss=8.8297 val_bpb=3.1201 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=10.9s
step=18 qat=enabled
step=20 train_loss=8.1036 train_bpb=2.8635 val_loss=8.2035 val_bpb=2.8988 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=22.8s
=== final_stats ===
steps=30
seconds=33.92
final_val_loss=8.2106
final_val_bpb=2.9013
compressed_model_size_bytes=16576687
code_size_bytes=29425
total_artifact_bytes=16606112
artifact_budget_ok=False
stats_path=runs/fineweb_16k_sample_promote_cpu/m4_d512_l4.json

=== sweep_ranking ===
1. m4_d512_l4 final_val_bpb=2.9013 params=19926656 artifact=16606112
2. m4_d448_l4 final_val_bpb=2.9547 params=17091312 artifact=14222955
3. m4_d384_l4 final_val_bpb=3.0431 params=14355552 artifact=12034880
4. m4_d320_l4 final_val_bpb=3.0711 params=11716880 artifact=9836247
results_path=runs/fineweb_16k_sample_promote_cpu/results.jsonl
```

Interpretation:

- The 16k tokenizer clearly beats the earlier 8k proxy runs.
- `d512_l4` remained the best metric point overall but was invalid.
- `d448_l4` emerged as the best valid 16k point from that CPU sweep, which made it the right anchor for a finer host-side MPS frontier search.

## Experiment 13. Host-Side MPS Frontier Sweep with `autopilot.py`

Status:

- Passed after one sandbox-only MPS failure

Purpose:

- Search the exact valid/invalid edge around the new 16k tokenizer setup while keeping durable logs on disk.

First attempt command:

```bash
MAX_STEPS=20 .venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_16k_sample/train --val-data-path ./data/tokens/fineweb_16k_sample/val --widths 448 464 480 496 512 --max-steps 20 --device mps --output-dir runs/fineweb_16k_frontier_mps --run-prefix fineweb16k
```

First attempt terminal failure:

```text
[1/5] run_id=fineweb16k_d448_l4 d_ff=1194
config: {'run_id': 'fineweb16k_d448_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 448, 'n_heads': 8, 'd_ff': 1194, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d448_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
RuntimeError: The MPS backend is supported on MacOS 14.0+. Current OS version can be queried using `sw_vers`
```

What we changed:

- Re-ran the exact same command outside the sandbox so the host MPS backend was available.

Successful terminal output:

```text
[1/5] run_id=fineweb16k_d448_l4 d_ff=1194
config: {'run_id': 'fineweb16k_d448_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 448, 'n_heads': 8, 'd_ff': 1194, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d448_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=17,091,312
step=0 train_loss=9.7302 train_bpb=3.4383 val_loss=9.7118 val_bpb=3.4318 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.0493 train_bpb=3.1977 val_loss=8.9785 val_bpb=3.1727 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=4.7s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=9.86
final_val_loss=8.8091
final_val_bpb=3.1128
compressed_model_size_bytes=14223410
code_size_bytes=29425
total_artifact_bytes=14252835
artifact_budget_ok=True
stats_path=runs/fineweb_16k_frontier_mps/fineweb16k_d448_l4.json
[2/5] run_id=fineweb16k_d464_l4 d_ff=1237
config: {'run_id': 'fineweb16k_d464_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 464, 'n_heads': 8, 'd_ff': 1237, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d464_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=17,791,268
step=0 train_loss=9.8371 train_bpb=3.4761 val_loss=9.8263 val_bpb=3.4723 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.0666 train_bpb=3.2038 val_loss=8.9936 val_bpb=3.1780 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=5.8s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=10.53
final_val_loss=8.8168
final_val_bpb=3.1155
compressed_model_size_bytes=14829703
code_size_bytes=29425
total_artifact_bytes=14859128
artifact_budget_ok=True
stats_path=runs/fineweb_16k_frontier_mps/fineweb16k_d464_l4.json
[3/5] run_id=fineweb16k_d480_l4 d_ff=1280
config: {'run_id': 'fineweb16k_d480_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 480, 'n_heads': 8, 'd_ff': 1280, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d480_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=18,497,400
step=0 train_loss=9.7873 train_bpb=3.4585 val_loss=9.7578 val_bpb=3.4480 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.0020 train_bpb=3.1810 val_loss=8.9222 val_bpb=3.1528 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=5.3s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=10.09
final_val_loss=8.8162
final_val_bpb=3.1153
compressed_model_size_bytes=15384433
code_size_bytes=29425
total_artifact_bytes=15413858
artifact_budget_ok=True
stats_path=runs/fineweb_16k_frontier_mps/fineweb16k_d480_l4.json
[4/5] run_id=fineweb16k_d496_l4 d_ff=1322
config: {'run_id': 'fineweb16k_d496_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 496, 'n_heads': 8, 'd_ff': 1322, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=19,208,220
step=0 train_loss=9.8014 train_bpb=3.4634 val_loss=9.7812 val_bpb=3.4563 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.0146 train_bpb=3.1854 val_loss=8.9263 val_bpb=3.1542 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=5.8s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=10.95
final_val_loss=8.8030
final_val_bpb=3.1106
compressed_model_size_bytes=15969941
code_size_bytes=29425
total_artifact_bytes=15999366
artifact_budget_ok=True
stats_path=runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.json
[5/5] run_id=fineweb16k_d512_l4 d_ff=1365
config: {'run_id': 'fineweb16k_d512_l4', 'data_path': './data/tokens/fineweb_16k_sample/train', 'val_data_path': './data/tokens/fineweb_16k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 512, 'n_heads': 8, 'd_ff': 1365, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'mps', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_16k_frontier_mps/fineweb16k_d512_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=mps
vocab_size=16384 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
avg_bytes_per_token=4.0828 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_16k_sample/val
parameters=19,926,656
step=0 train_loss=9.7808 train_bpb=3.4562 val_loss=9.7531 val_bpb=3.4464 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=8.9222 train_bpb=3.1528 val_loss=8.8297 val_bpb=3.1201 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=4.9s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=9.80
final_val_loss=8.7164
final_val_bpb=3.0801
compressed_model_size_bytes=16558095
code_size_bytes=29425
total_artifact_bytes=16587520
artifact_budget_ok=False
stats_path=runs/fineweb_16k_frontier_mps/fineweb16k_d512_l4.json

=== autopilot_ranking ===
1. fineweb16k_d512_l4 final_val_bpb=3.0801 params=19926656 artifact=16587520
2. fineweb16k_d496_l4 final_val_bpb=3.1106 params=19208220 artifact=15999366
3. fineweb16k_d448_l4 final_val_bpb=3.1128 params=17091312 artifact=14252835
4. fineweb16k_d480_l4 final_val_bpb=3.1153 params=18497400 artifact=15413858
5. fineweb16k_d464_l4 final_val_bpb=3.1155 params=17791268 artifact=14859128

=== autopilot_under_budget_ranking ===
1. fineweb16k_d496_l4 final_val_bpb=3.1106 params=19208220 artifact=15999366
2. fineweb16k_d448_l4 final_val_bpb=3.1128 params=17091312 artifact=14252835
3. fineweb16k_d480_l4 final_val_bpb=3.1153 params=18497400 artifact=15413858
4. fineweb16k_d464_l4 final_val_bpb=3.1155 params=17791268 artifact=14859128
results_path=runs/fineweb_16k_frontier_mps/results.jsonl
leaderboard_path=runs/fineweb_16k_frontier_mps/leaderboard.json
```

Interpretation:

- The best valid 16k point is now `fineweb16k_d496_l4` at `final_val_bpb=3.1106`.
- It fits with only `634` bytes of headroom, so this is a true budget-edge result rather than a loose under-budget point.
- `d512_l4` is still better numerically, but it remains invalid.
- The non-monotonic ordering (`d496` better than `d480` and `d464`, while `d448` stays competitive) suggests we are in the noisy edge regime and should compare tokenizer size next rather than only keep widening.

## Current Working Hypothesis

The strongest local submission candidate we have right now is:

1. `vocab_size=16384`
2. `d_model=496`
3. `n_loops=4`
4. `n_heads=8`
5. `d_ff=1322`

Reasoning:

- It is the best under-budget real-data result we have actually logged.
- It sits almost exactly on the size cap, which is where we want to be.
- The 16k tokenizer clearly helped, so the next likely improvement is another tokenizer-size jump paired with a narrower width search.

What still keeps this far from first place:

- The official tracked score on March 18, 2026 is `1.22436570`.
- Our current local proxy is still much worse because we are using tiny local data slices and very short local runs.
- These experiments are still useful because they help us choose architecture and tokenizer directions before expensive full-scale training.

## Next Command To Run

This is the next command we want to execute:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 24576 --output-dir ./data/tokenizers --prefix fineweb_24k_sample && .venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_24k_sample --output-dir ./data/tokens/fineweb_24k_sample && .venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_24k_sample/train --val-data-path ./data/tokens/fineweb_24k_sample/val --widths 352 384 416 448 --max-steps 20 --output-dir runs/fineweb_24k_frontier_cpu --run-prefix fineweb24k
```

Why this exact command:

- The 16k tokenizer improved the metric, so the next thing to test is whether a 24k tokenizer gives us another denominator win.
- We deliberately shrink the width range because the 16k `d496` result already uses almost the full artifact budget.
- This is the cleanest next comparison that can still run locally without adding more external dependencies or dataset churn.

## Experiment 14. 24k Tokenizer Frontier Rejection Test

Status:

- Passed

Purpose:

- Check whether a larger tokenizer than 16k can win overall if we compensate with a narrower model.

Command:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 24576 --output-dir ./data/tokenizers --prefix fineweb_24k_sample && .venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_24k_sample --output-dir ./data/tokens/fineweb_24k_sample && .venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_24k_sample/train --val-data-path ./data/tokens/fineweb_24k_sample/val --widths 352 384 416 448 --max-steps 20 --output-dir runs/fineweb_24k_frontier_cpu --run-prefix fineweb24k
```

Terminal output:

```text
source=local-files
vocab_path=data/tokenizers/fineweb_24k_sample-vocab.json
merges_path=data/tokenizers/fineweb_24k_sample-merges.txt
raw_size_bytes=598029
compressed_size_bytes=249410
{
  "source": "local-files",
  "output_dir": "data/tokens/fineweb_24k_sample",
  "tokenizer_prefix": "./data/tokenizers/fineweb_24k_sample",
  "train": {
    "split": "train",
    "source": "local-files",
    "tokenizer_prefix": "./data/tokenizers/fineweb_24k_sample",
    "vocab_size": 24576,
    "token_dtype": "uint16",
    "docs": 15000,
    "shards": 6,
    "total_bytes": 46410649,
    "total_tokens": 10767318,
    "avg_bytes_per_token": 4.310325839730934
  },
  "val": {
    "split": "val",
    "source": "local-files",
    "tokenizer_prefix": "./data/tokenizers/fineweb_24k_sample",
    "vocab_size": 24576,
    "token_dtype": "uint16",
    "docs": 1000,
    "shards": 1,
    "total_bytes": 3032792,
    "total_tokens": 713604,
    "avg_bytes_per_token": 4.249964966564089
  }
}
[1/4] run_id=fineweb24k_d352_l4
config: {'run_id': 'fineweb24k_d352_l4', 'data_path': './data/tokens/fineweb_24k_sample/train', 'val_data_path': './data/tokens/fineweb_24k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 352, 'n_heads': 8, 'd_ff': 938, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_24k_frontier_cpu/fineweb24k_d352_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=24576 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
avg_bytes_per_token=4.2500 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
parameters=18,790,552
step=0 train_loss=10.2081 train_bpb=3.4652 val_loss=10.1993 val_bpb=3.4622 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.5744 train_bpb=3.2501 val_loss=9.5415 val_bpb=3.2389 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=9.8s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=20.51
final_val_loss=9.4388
final_val_bpb=3.2041
compressed_model_size_bytes=15619565
code_size_bytes=29425
total_artifact_bytes=15648990
artifact_budget_ok=True
stats_path=runs/fineweb_24k_frontier_cpu/fineweb24k_d352_l4.json
[2/4] run_id=fineweb24k_d384_l4
config: {'run_id': 'fineweb24k_d384_l4', 'data_path': './data/tokens/fineweb_24k_sample/train', 'val_data_path': './data/tokens/fineweb_24k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 384, 'n_heads': 8, 'd_ff': 1024, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_24k_frontier_cpu/fineweb24k_d384_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=24576 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
avg_bytes_per_token=4.2500 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
parameters=20,647,008
step=0 train_loss=10.1246 train_bpb=3.4369 val_loss=10.1068 val_bpb=3.4309 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.4617 train_bpb=3.2119 val_loss=9.4457 val_bpb=3.2064 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=10.2s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=20.48
final_val_loss=9.2253
final_val_bpb=3.1316
compressed_model_size_bytes=17126552
code_size_bytes=29425
total_artifact_bytes=17155977
artifact_budget_ok=False
stats_path=runs/fineweb_24k_frontier_cpu/fineweb24k_d384_l4.json
[3/4] run_id=fineweb24k_d416_l4
config: {'run_id': 'fineweb24k_d416_l4', 'data_path': './data/tokens/fineweb_24k_sample/train', 'val_data_path': './data/tokens/fineweb_24k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 416, 'n_heads': 8, 'd_ff': 1109, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_24k_frontier_cpu/fineweb24k_d416_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=24576 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
avg_bytes_per_token=4.2500 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
parameters=22,526,920
step=0 train_loss=10.1504 train_bpb=3.4457 val_loss=10.1389 val_bpb=3.4418 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.3998 train_bpb=3.1909 val_loss=9.3670 val_bpb=3.1797 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=11.3s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=23.08
final_val_loss=9.1357
final_val_bpb=3.1012
compressed_model_size_bytes=18800770
code_size_bytes=29425
total_artifact_bytes=18830195
artifact_budget_ok=False
stats_path=runs/fineweb_24k_frontier_cpu/fineweb24k_d416_l4.json
[4/4] run_id=fineweb24k_d448_l4
config: {'run_id': 'fineweb24k_d448_l4', 'data_path': './data/tokens/fineweb_24k_sample/train', 'val_data_path': './data/tokens/fineweb_24k_sample/val', 'token_dtype': None, 'vocab_size': None, 'd_model': 448, 'n_heads': 8, 'd_ff': 1194, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 8192, 'val_batch_tokens': 8192, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 20, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': None, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': '', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': 'runs/fineweb_24k_frontier_cpu/fineweb24k_d448_l4.json'}
train_source=6 shard(s) val_source=1 shard(s) device=cpu
vocab_size=24576 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
token_dtype=uint16 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
avg_bytes_per_token=4.2500 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_24k_sample/val
parameters=24,431,344
step=0 train_loss=10.2190 train_bpb=3.4689 val_loss=10.1987 val_bpb=3.4621 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=9.4437 train_bpb=3.2058 val_loss=9.4132 val_bpb=3.1954 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=12.3s
step=12 qat=enabled
=== final_stats ===
steps=20
seconds=25.31
final_val_loss=9.1660
final_val_bpb=3.1115
compressed_model_size_bytes=20393176
code_size_bytes=29425
total_artifact_bytes=20422601
artifact_budget_ok=False
stats_path=runs/fineweb_24k_frontier_cpu/fineweb24k_d448_l4.json

=== autopilot_ranking ===
1. fineweb24k_d416_l4 final_val_bpb=3.1012 params=22526920 artifact=18830195
2. fineweb24k_d448_l4 final_val_bpb=3.1115 params=24431344 artifact=20422601
3. fineweb24k_d384_l4 final_val_bpb=3.1316 params=20647008 artifact=17155977
4. fineweb24k_d352_l4 final_val_bpb=3.2041 params=18790552 artifact=15648990

=== autopilot_under_budget_ranking ===
1. fineweb24k_d352_l4 final_val_bpb=3.2041 params=18790552 artifact=15648990
results_path=runs/fineweb_24k_frontier_cpu/results.jsonl
leaderboard_path=runs/fineweb_24k_frontier_cpu/leaderboard.json
```

Interpretation:

- The 24k tokenizer improved bytes/token again, but it made the model family much more expensive.
- The only valid 24k point in this band was `d352_l4`, and it was much worse than the 16k best.
- The best 24k metric point was `d416_l4`, but it was badly over budget at `18,830,195` bytes.
- This is strong evidence that 24k is the wrong next step for this architecture family.

## Current Working Hypothesis

The best direction right now is no longer "keep increasing tokenizer size." It is:

1. keep the tokenizer around `16k`
2. keep the model near the `d496_l4` budget edge
3. search intermediate tokenizer sizes only if they are close to 16k, not as large as 24k
4. spend more effort on model-side efficiency and training quality instead of just pushing vocab upward

Reasoning:

- `16k` clearly beat `8k`.
- `24k` did not produce a better valid point.
- That suggests the optimum for this local setup is somewhere between `16k` and `24k`, or that `16k` is already near the sweet spot.

## Next Command To Run

This is the next command we want to execute:

```bash
.venv/bin/python train_tokenizer.py --input-file ./data/raw/fineweb_16k_sample/train.jsonl --vocab-size 20480 --output-dir ./data/tokenizers --prefix fineweb_20k_sample && .venv/bin/python prepare_tokens.py --train-input-file ./data/raw/fineweb_16k_sample/train.jsonl --val-input-file ./data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_20k_sample --output-dir ./data/tokens/fineweb_20k_sample && .venv/bin/python autopilot.py --data-path ./data/tokens/fineweb_20k_sample/train --val-data-path ./data/tokens/fineweb_20k_sample/val --widths 416 448 480 --max-steps 20 --output-dir runs/fineweb_20k_frontier_cpu --run-prefix fineweb20k
```

Why this exact command:

- `20k` is the natural midpoint after learning that `16k` helps but `24k` is too large.
- The width band starts at `416` because the 24k valid point at `352` was too weak, while `16k` worked all the way up to `496`.
- This is the cleanest way to test whether the real sweet spot is between the two tokenizer sizes we have already checked.

## Experiment 15. Official Repository Audit and Rules Alignment

Status:

- Passed

Purpose:

- Review the official OpenAI Parameter Golf repository before continuing experiments, so local work stays aligned with the real rules, validation setup, and submission format.

Command:

```bash
tmpdir=$(mktemp -d /tmp/parameter-golf-official.XXXXXX) && git clone --depth 1 https://github.com/openai/parameter-golf "$tmpdir" && printf '%s\n' "$tmpdir"
```

Supporting inspection commands:

```bash
sed -n '1,260p' /tmp/parameter-golf-official.BbMmsz/README.md
sed -n '1,260p' /tmp/parameter-golf-official.BbMmsz/data/README.md
sed -n '1,240p' /tmp/parameter-golf-official.BbMmsz/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
cat /tmp/parameter-golf-official.BbMmsz/records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json
wc -l /tmp/parameter-golf-official.BbMmsz/train_gpt.py /tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
git remote -v
```

Terminal output:

```text
Cloning into '/tmp/parameter-golf-official.BbMmsz'...
/tmp/parameter-golf-official.BbMmsz
{
  "author": "Baseline",
  "github_id": "openai",
  "name": "Naive Baseline",
  "blurb": "SP-1024 9x512 KV4 run on pgut1 using the published Hugging Face fineweb10B_sp1024 export and the current train_gpt.py; score is the default final int8+zlib roundtrip metric under the 16,000,000-byte cap.",
  "date": "2026-03-18T14:56:29Z",
  "val_loss": 2.07269931,
  "val_bpb": 1.2243657,
  "bytes_total": 15863489,
  "bytes_code": 47642
}
    1126 /tmp/parameter-golf-official.BbMmsz/train_gpt.py
    1088 /tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
    2214 total
origin	https://github.com/aryan-cs/parameter-golf.git (fetch)
origin	https://github.com/aryan-cs/parameter-golf.git (push)
```

What we learned:

- The real record track is lowest `val_bpb` under a decimal `16,000,000` byte cap and under `10 minutes` on `8xH100`.
- The official validation set is the fixed full `fineweb_val_*` split, i.e. the first `50,000` documents.
- The official FAQ explicitly forbids external downloads, training-data access, and network calls during evaluation.
- Record submissions must be packaged as a new folder under `records/track_10min_16mb/` with `README.md`, `submission.json`, `train.log`, and a runnable `train_gpt.py`.
- The root official `train_gpt.py` and `train_gpt_mlx.py` are starter scripts, not where final competitive submissions should live.
- The current public #1 remains the `Naive Baseline` at exact `val_bpb=1.22436570`.

What changed in our repo:

- Added `RULES.md` as the local summary of the official competition rules and source links.
- Updated `PLAN.md` to point at `RULES.md` and corrected the deadline year.
- Committed that rules alignment as `888ddf0` with message `Document official competition rules and align plan`.

Interpretation:

- Our current local sweep workflow is still useful, but only as a proxy search loop.
- We should stop treating local sample-slice scores as leaderboard-like.
- The next high-leverage step is to align our workflow with the official data, validation, and `records/...` packaging path before spending more time on tokenizer-width frontier tuning.

## Current Working Hypothesis

The highest-value direction has changed slightly after the official audit:

1. keep using the current local pipeline for fast proxy search
2. label every proxy result clearly as non-official
3. shift the next engineering work toward official-path alignment
4. only resume deeper frontier search once we can package and validate runs in a competition-shaped way

Reasoning:

- We now know exactly what OpenAI will inspect for real submissions.
- Our current local setup does not yet match the official validation and records flow.
- Better local proxy scores alone do not help if the eventual submission path is mismatched.

## Next Command To Run

This is the next command we want to execute:

```bash
git push origin master
```

Why this exact command:

- The repo now has three important checkpoints that should be on the remote:
  - `69f2c93` `Log 24k tokenizer rejection experiment`
  - `888ddf0` `Document official competition rules and align plan`
- Keeping the remote current makes the audit trail durable and keeps the working branch in sync as we continue.
