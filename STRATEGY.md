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

As of March 19, 2026, the official repo leaderboard still shows:

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
- source: official `Naive Baseline` record, rechecked on March 19, 2026
- official record-claim significance margin: `0.005 nats = 0.007213475204444817 bpb`
- practical record-claim target: `1.217152224795555` bpb or better

Current interesting non-record comparison:

- exact `val_bpb`: `1.20737944`
- source: official `4-Hour Baseline` non-record run, rechecked on March 19, 2026

Our current best local exact result:

- exact `final_val_bpb`: `1.6238459688827054`
- short form: `1.6238`
- run id: `bytelevel24k_d512_gqa_softcap_s3200`
- artifact: `21,566,256` bytes
- status: exact ByteLevel BPE on the local `24k` sample-data path; strong local milestone, but not competition-valid and not artifact-valid yet

Current best official-tokenizer local exact-bpb result:

- exact sampled `best_val_bpb`: `1.995522745133331`
- short form: `1.9955`
- run id: `torch_sp1024_d512_l4_b64k_s200`
- artifact: `3,566,778` bytes
- status: exact SentencePiece byte accounting on sampled validation batches (`VAL_STEPS=16`), still a local proxy until we run the fixed full validation split
- interpretation guardrail: treat sampled exact validation as roughly `+/- 0.05` bpb until confirmed on the full official validation pass

Current best long-context (`SEQ_LEN=1024`) official-tokenizer result:

- exact sampled `best_val_bpb`: `2.0050325515775427`
- short form: `2.0050`
- run id: `torch_sp1024_d512_l4_b32k_s400_seq1024_cd20`
- artifact: `3,581,567` bytes
- status: best long-context branch so far; only `0.009509806444211607` bpb worse than the overall sampled best, which is well inside the local sampled-eval noise band

Current best exact `24k` ByteLevel result:

- exact sampled `best_val_bpb`: `1.6238459688827054`
- short form: `1.6238`
- run id: `bytelevel24k_d512_gqa_softcap_s3200`
- artifact: `21,566,256` bytes
- status: best exact large-tokenizer result so far; still far above the artifact cap, but now the strongest overall local exact score in the repo

Most recent stopped frontier checkpoint:

- run id: `bytelevel24k_d512_gqa_softcap_relu2_core_s1600`
- latest checkpoint: `step=800`
- exact `val_bpb`: `1.8177`
- gap to local `1.5`: `0.3177`
- status: stopped early after `step=800`; the simplified `relu2` branch still trailed the older plain `24k` reference (`1.7954`), so the `relu2` family is no longer the main active hypothesis

Current live frontier checkpoint:

- run id: `bytelevel24k_d640_gqa_softcap_s1600`
- latest checkpoint: `step=800`
- live `val_bpb`: `1.7533`
- gap to local `1.5`: `0.2533`
- status: now active on MPS as the width-upscaled version of the proven plain `24k` GQA/softcap recipe

Current prepared next-tokenizer branch:

- tokenizer: `fineweb_32k_sample`
- train `avg_bytes_per_token`: `4.4202857706695236`
- val `avg_bytes_per_token`: `4.351551413669658`
- status: prepared on disk and ready for model training

Prepared tokenizer ladder behind the live branch:

- `fineweb_48k_sample`
  - train `avg_bytes_per_token`: `4.550211732381878`
  - val `avg_bytes_per_token`: `4.466813263206576`
- `fineweb_64k_sample`
  - train `avg_bytes_per_token`: `4.62566836285424`
  - val `avg_bytes_per_token`: `4.534385498540016`
- status: `48k` is now a rejected local branch; `64k` remains prepared only as a contingency, not the current main line

Current prepared next-architecture branch:

- knobs: width-upscaled plain `24k` variants such as `D_MODEL=640`
- status: now upgraded from "prepared" to "live exact run" on the `24k` tokenizer family
- rationale: the tokenizer ladder and `relu2` block variants both stalled, so the next clean lever is more capacity on the strongest plain branch

Gap versus official main-track leader for our best exact sampled official-tokenizer run:

- `1.995522745133331 - 1.22436570 = 0.7711570451333312` bpb worse
- `0.7711570451333312 / log2(e) = 0.534525331603107` nats worse

Gap versus official main-track leader for our best tokenizer-changed local exact run:

- `1.6238459688827054 - 1.22436570 = 0.3994802688827055` bpb worse
- `0.3994802688827055 / log2(e) = 0.27689246300965436` nats worse

Gap versus the immediate `1.5` local target:

- `1.6238459688827054 - 1.5 = 0.12384596888270538` bpb worse

PR opening rule:

- Open a main-track PR only when we have an exact competition-valid score below `1.22436570`
- Treat `1.217152224795555` as the real threshold for a clean new-record claim under the official `0.005 nats` bar
- And the run must also satisfy all official submission conditions:
  - under `16,000,000` bytes total artifact size
  - reproducible under `10 minutes` on `8xH100`
  - official-style validation on the fixed full `fineweb_val_*` split
  - record folder packaging under `records/track_10min_16mb/`
  - enough evidence for the official significance and reproducibility requirements

Sampled-eval policy:

- Do not make architecture decisions from local sampled exact validation deltas smaller than about `0.05` bpb
- Label sampled exact results as proxies until we run a full fixed-split validation pass

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

### 8. Add Official-Shard Compatibility and Record Packaging

Commit:

- `5b10c5a` `Add official shard support and record packager`

Purpose:

- Close two more gaps between our local proxy workflow and the official submission path.

Key code from `train_gpt.py`:

```python
if split in {"train", "val"}:
    split_pattern = f"fineweb_{split}_*.bin"
    candidates.extend(sorted(root.glob(split_pattern)))
```

Key code for official headered `.bin` support:

```python
if header.size >= 3 and int(header[0]) == OFFICIAL_DATAFILE_MAGIC:
    token_count = int(header[2])
    return np.memmap(path, dtype=dtype, mode="r", offset=OFFICIAL_DATAFILE_HEADER_BYTES, shape=(token_count,))
```

Key code from `package_record.py`:

```python
payload = {
    "author": args.author,
    "github_id": args.github_id,
    "name": args.name,
    "blurb": args.blurb,
    "date": utc_now_iso(),
    "val_loss": stats["final_val_loss"],
    "val_bpb": stats["final_val_bpb"],
    "bytes_total": stats["total_artifact_bytes"],
    "bytes_code": stats["code_size_bytes"],
}
```

Why this matters:

- Our trainer can now ingest the official shard format instead of only our own packed-sample format.
- We can now generate official-style `records/...` folders directly from a finished run.
- This still does not make our current runs competition-valid, but it removes two concrete workflow blockers.

### 9. Add Official MLX Run Wrapper

Commit:

- `fd6e1ce` `Add official MLX run wrapper`

Purpose:

- Automate official `train_gpt_mlx.py` launches, parse exact final `val_bpb`, and save machine-readable summaries beside the logs.

Key code from `run_official_mlx.py`:

```python
EXACT_METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
```

Key code for persisted parsed summaries:

```python
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
```

Why this matters:

- We no longer have to hand-scrape official MLX logs to know whether a run actually improved.
- This is the first local automation layer aimed directly at official-path runs rather than only local proxy sweeps.
- It also keeps the repo clean by ignoring transient `logs/` output.

### 10. Add Exact SentencePiece `val_bpb` to the Fast Trainer

Commit:

- `a1ff57b` `Add exact SentencePiece val_bpb support`

Purpose:

- Make the faster PyTorch/MPS trainer capable of tokenizer-aware byte accounting instead of only the old average-bytes approximation.

Key code from `train_gpt.py`:

```python
def build_sentencepiece_bpb_helper(tokenizer_path: str, vocab_size: int) -> SentencePieceBPBHelper:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
```

Key code for exact full-scan validation:

```python
if cfg.val_steps <= 0:
    for arr in val_arrays:
        total_seqs = (len(arr) - 1) // cfg.seq_len
```

Why this matters:

- We now have a much faster local loop that still respects the official tokenizer's byte semantics.
- `VAL_STEPS=0` now means "scan the validation shards sequentially" instead of falling back to a sampled estimate.
- This is the first serious path in the repo that combines official shard ingestion, official tokenizer semantics, and MPS-friendly training speed.

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

## Experiment 16. Official Shard Ingestion Smoke Test

Status:

- Passed

Purpose:

- Verify that our local trainer can ingest the official OpenAI challenge shard layout and the official headered `.bin` file format.

Official data download command:

```bash
.venv/bin/python /tmp/parameter-golf-official.BbMmsz/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Supporting command to estimate the SP-1024 bytes/token constant from the official baseline metadata:

```bash
.venv/bin/python -c "import math; print(2.07269931*math.log2(math.e)/1.22436570)"
```

Ingestion smoke command:

```bash
DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 SEQ_LEN=128 TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 MAX_STEPS=1 DEVICE=cpu .venv/bin/python train_gpt.py
```

Terminal output:

```text
2.442303811509075
fineweb_train_000000.bin
fineweb_val_000000.bin
config: {'run_id': 'dev_smoke', 'data_path': '/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024', 'val_data_path': '/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024', 'token_dtype': 'uint16', 'vocab_size': 1024, 'd_model': 256, 'n_heads': 8, 'd_ff': 682, 'n_loops': 4, 'seq_len': 128, 'train_batch_tokens': 1024, 'val_batch_tokens': 1024, 'val_steps': 4, 'val_loss_every': 10, 'max_steps': 1, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': 2.442303811509075, 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'cpu', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': ''}
train_source=1 shard(s) val_source=1 shard(s) device=cpu
vocab_size=1024 source=env
token_dtype=uint16 source=env
avg_bytes_per_token=2.4423 source=env
parameters=1,312,320
step=0 train_loss=6.9550 train_bpb=4.1084 val_loss=6.8524 val_bpb=4.0478 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=0.14
final_val_loss=6.8386
final_val_bpb=4.0397
compressed_model_size_bytes=1136396
code_size_bytes=31140
total_artifact_bytes=1167536
artifact_budget_ok=True
```

Interpretation:

- Our trainer now correctly distinguishes official `fineweb_train_*.bin` and `fineweb_val_*.bin` files inside a single official dataset directory.
- It also correctly skips the official binary header before reading token payloads.
- This is an infrastructure validation only, not a leaderboard-relevant score.

## Experiment 17. Official-Style Record Packaging Smoke Test

Status:

- Passed

Purpose:

- Verify that we can turn a finished run into the official `records/...` folder shape required for eventual PRs.

Command:

```bash
tmpout=$(mktemp -d /tmp/golf-records-smoke.XXXXXX) && .venv/bin/python package_record.py --stats runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.json --log runs/fineweb_16k_frontier_mps/fineweb16k_d496_l4.log --name "FineWeb 16k d496 l4" --slug fineweb16k_d496_l4 --author "Aryan" --github-id aryan-cs --blurb "Local proxy run packaged for records-folder iteration." --track-dir track_non_record_16mb --output-root "$tmpout"
```

Terminal output:

```text
record_dir=/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4
submission_path=/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4/submission.json
/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4/README.md
/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4/submission.json
/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4/train.log
/tmp/golf-records-smoke.VKgaq3/track_non_record_16mb/2026-03-18_fineweb16k_d496_l4/train_gpt.py
```

Generated `submission.json` payload:

```json
{
  "author": "Aryan",
  "github_id": "aryan-cs",
  "name": "FineWeb 16k d496 l4",
  "blurb": "Local proxy run packaged for records-folder iteration.",
  "date": "2026-03-18T20:14:36Z",
  "val_loss": 8.803004264831543,
  "val_bpb": 3.1106495880562357,
  "bytes_total": 15999366,
  "bytes_code": 29425,
  "best_val_loss": 8.803004264831543,
  "best_val_bpb": 3.1106495880562357,
  "bytes_model_int8_zlib": 15969941,
  "steps": 20,
  "wallclock_seconds": 10.948360749986023
}
```

Interpretation:

- We can now generate the official record-folder file set automatically:
  - `README.md`
  - `submission.json`
  - `train.log`
  - `train_gpt.py`
- The generated README is intentionally conservative and should be edited before any real submission.
- This removes another manual bottleneck between “good run” and “PR-ready record folder.”

## Experiment 18. Official MLX Smoke on Real `sp1024` Data

Status:

- Aborted manually after step `200/200`

Purpose:

- Verify that the official Apple Silicon training path runs end-to-end against the real cached challenge data and tokenizer.

Command:

```bash
RUN_ID=official_mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model .venv/bin/python /tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
```

Observed terminal output from the log:

```text
run_id:official_mlx_smoke
mlx_version:0.31.1
train_loader:shards pattern=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
val_loader:shards pattern=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632
WARNING: train_loader:subset dataset:fineweb10B_sp1024 train_shards:1/195 new epochs will arrive sooner than the full dataset
model_params:17059912 vocab_size:1024 layers:9 dim:512 heads:8 kv_heads:4 seq_len:1024 tie_embeddings:True
iterations:200 train_batch_tokens:8192 grad_accum_steps:8 microbatch_tokens:1024 microbatch_batch_size:1 val_batch_size:8192 warmup_steps:20 max_wallclock_seconds:600.000
step:1/200 train_loss:6.9428 train_time:147ms step_avg:146.52ms tok_s:55916
step:10/200 train_loss:6.3728 train_time:4485ms step_avg:448.47ms tok_s:17440
step:200/200 train_loss:3.8992 train_time:96143ms step_avg:480.71ms tok_s:17087
```

Process state before manual stop:

```text
19708 05:06  92.1  6.3 Rs   .venv/bin/python /tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
```

Relevant code that explained the slowdown:

```python
val_batch_tokens = args.val_batch_size // args.grad_accum_steps
```

Interpretation:

- The run itself was healthy and finished all `200` train steps.
- The real problem was evaluation geometry, not model stability.
- With `VAL_BATCH_SIZE=8192` and default `GRAD_ACCUM_STEPS=8`, the official script evaluated only `1024` tokens per validation chunk across the full `62,021,632`-token validation set.
- That made the final exact metric path impractically slow on the Mac, so we stopped it and treated it as a negative-result workflow lesson rather than waiting indefinitely for a low-value smoke score.

## Experiment 19. Expand Official `sp1024` Cache from 1 to 10 Train Shards

Status:

- Passed

Purpose:

- Give the next official-path runs enough training data to matter.

Commands:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024')
files = sorted(root.glob('fineweb_train_*.bin'))
print({'train_shards': len(files), 'first': files[0].name if files else None, 'last': files[-1].name if files else None})
PY

.venv/bin/python /tmp/parameter-golf-official.BbMmsz/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024')
files = sorted(root.glob('fineweb_train_*.bin'))
print({'train_shards': len(files), 'first': files[0].name if files else None, 'last': files[-1].name if files else None})
PY
```

Terminal output:

```text
{'train_shards': 1, 'first': 'fineweb_train_000000.bin', 'last': 'fineweb_train_000000.bin'}
{'train_shards': 10, 'first': 'fineweb_train_000000.bin', 'last': 'fineweb_train_000009.bin'}
```

Interpretation:

- We now have `10` official train shards locally instead of `1`.
- That reduces the chance that our next local official-path run just overfits or cycles through the same tiny prefix too quickly.

## Experiment 20. Baseline-Scale Official MLX Attempt on 10 Shards

Status:

- Failed with `SIGKILL`

Purpose:

- Try the closest local approximation to the official baseline geometry that still seemed plausible on a 25GB Apple Silicon machine.

Command:

```bash
.venv/bin/python run_official_mlx.py --official-root /tmp/parameter-golf-official.BbMmsz --run-id official_mlx_s10_i600_tb524k_mb32k --output-dir runs/official_mlx_sp1024 --data-path /tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 --tokenizer-path /tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model --env ITERATIONS=600 --env TRAIN_BATCH_TOKENS=524288 --env VAL_BATCH_SIZE=524288 --env TRAIN_LOG_EVERY=50 --env MAX_WALLCLOCK_SECONDS=0 --env MLX_MAX_MICROBATCH_TOKENS=32768
```

Terminal output:

```text
run_id=official_mlx_s10_i600_tb524k_mb32k
log_path=/Users/aryan/Desktop/golf/runs/official_mlx_sp1024/official_mlx_s10_i600_tb524k_mb32k.txt
command=/Users/aryan/Desktop/golf/.venv/bin/python /private/tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
output_dir=/Users/aryan/Desktop/golf/runs/official_mlx_sp1024
run_id:official_mlx_s10_i600_tb524k_mb32k
mlx_version:0.31.1
train_loader:shards pattern=/private/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
val_loader:shards pattern=/private/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632
WARNING: train_loader:subset dataset:fineweb10B_sp1024 train_shards:10/195 new epochs will arrive sooner than the full dataset
iterations:600 train_batch_tokens:524288 grad_accum_steps:8 microbatch_tokens:65536 microbatch_batch_size:64 val_batch_size:524288 warmup_steps:20 max_wallclock_seconds:0.000
mlx_max_microbatch_tokens:32768
Traceback (most recent call last):
  File "/Users/aryan/Desktop/golf/run_official_mlx.py", line 178, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "/Users/aryan/Desktop/golf/run_official_mlx.py", line 144, in main
    stream_process(command, env)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/Users/aryan/Desktop/golf/run_official_mlx.py", line 110, in stream_process
    raise subprocess.CalledProcessError(return_code, command)
subprocess.CalledProcessError: Command '['/Users/aryan/Desktop/golf/.venv/bin/python', '/private/tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py']' died with <Signals.SIGKILL: 9>.
```

Interpretation:

- This batch shape was too aggressive for the local machine.
- The process died before warmup logging began, which strongly suggests memory pressure during compile or first graph materialization.
- The right move was not to abandon official-path runs, but to preserve roughly the same total token budget with a smaller effective batch and more iterations.

## Experiment 21. Safer Official MLX Medium-Batch Run on 10 Shards

Status:

- Stopped after proving memory stability but rejected as the main local loop because throughput was too low

Purpose:

- Preserve roughly the same total token target as the failed baseline-scale attempt while dropping memory pressure enough to survive on the Mac.

Command:

```bash
.venv/bin/python run_official_mlx.py --official-root /tmp/parameter-golf-official.BbMmsz --run-id official_mlx_s10_i1200_tb262k_mb16k_logit8k --output-dir runs/official_mlx_sp1024 --data-path /tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 --tokenizer-path /tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model --env ITERATIONS=1200 --env TRAIN_BATCH_TOKENS=262144 --env VAL_BATCH_SIZE=262144 --env TRAIN_LOG_EVERY=100 --env MAX_WALLCLOCK_SECONDS=0 --env MLX_MAX_MICROBATCH_TOKENS=16384 --env LOGIT_CHUNK_TOKENS=8192
```

Terminal output:

```text
run_id=official_mlx_s10_i1200_tb262k_mb16k_logit8k
log_path=/Users/aryan/Desktop/golf/runs/official_mlx_sp1024/official_mlx_s10_i1200_tb262k_mb16k_logit8k.txt
command=/Users/aryan/Desktop/golf/.venv/bin/python /private/tmp/parameter-golf-official.BbMmsz/train_gpt_mlx.py
output_dir=/Users/aryan/Desktop/golf/runs/official_mlx_sp1024
run_id:official_mlx_s10_i1200_tb262k_mb16k_logit8k
mlx_version:0.31.1
train_loader:shards pattern=/private/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
val_loader:shards pattern=/private/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632
WARNING: train_loader:subset dataset:fineweb10B_sp1024 train_shards:10/195 new epochs will arrive sooner than the full dataset
iterations:1200 train_batch_tokens:262144 grad_accum_steps:8 microbatch_tokens:32768 microbatch_batch_size:32 val_batch_size:262144 warmup_steps:20 max_wallclock_seconds:0.000
mlx_max_microbatch_tokens:16384
warmup_step:1/20
warmup_step:2/20
```

Follow-up relaunch with warmups disabled:

```text
run_id=official_mlx_s10_i1000_tb262k_mb16k_logit8k_w0
iterations:1000 train_batch_tokens:262144 grad_accum_steps:8 microbatch_tokens:32768 microbatch_batch_size:32 val_batch_size:262144 warmup_steps:0 max_wallclock_seconds:0.000
step:1/1000 train_loss:6.9380 train_time:75920ms step_avg:75919.79ms tok_s:3453
```

Interpretation:

- This run has not been killed, which is already better than the previous baseline-scale attempt.
- The configuration is memory-stable on the machine, so we learned that `262144` train tokens and `16384` microbatch chunks are feasible.
- But the throughput was far too low for the MLX path to be the main local search loop.
- That is why we pivoted to adding exact SentencePiece `val_bpb` support to the faster MPS trainer.

## Experiment 22. Tiny Official-Shard Smoke for Exact SentencePiece `val_bpb`

Status:

- Passed

Purpose:

- Prove that the new `TOKENIZER_PATH` plus `VAL_STEPS=0` path in `train_gpt.py` really executes exact SentencePiece-aware byte accounting on official-format shards.

Command:

```bash
tmpdir=$(.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import tempfile
MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
root = Path('/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024')
out = Path(tempfile.mkdtemp(prefix='golf-sp-bpb-smoke.', dir='/tmp'))
for split in ('train', 'val'):
    src = root / f'fineweb_{split}_000000.bin'
    token_count = 8193
    tokens = np.fromfile(src, dtype='<u2', count=token_count, offset=HEADER_INTS * 4)
    new_header = np.zeros((HEADER_INTS,), dtype='<i4')
    new_header[0] = MAGIC
    new_header[1] = VERSION
    new_header[2] = token_count
    dst = out / f'fineweb_{split}_000000.bin'
    with dst.open('wb') as handle:
        handle.write(new_header.tobytes())
        handle.write(tokens.tobytes())
print(out)
PY
) && TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH="$tmpdir" VAL_DATA_PATH="$tmpdir" VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=64 N_HEADS=4 D_FF=170 N_LOOPS=2 SEQ_LEN=128 TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 VAL_STEPS=0 VAL_LOSS_EVERY=0 MAX_STEPS=1 DEVICE=cpu .venv/bin/python train_gpt.py
```

Terminal output:

```text
config: {'run_id': 'dev_smoke', 'data_path': '/tmp/golf-sp-bpb-smoke._e9mz3rg', 'val_data_path': '/tmp/golf-sp-bpb-smoke._e9mz3rg', 'token_dtype': 'uint16', 'vocab_size': 1024, 'd_model': 64, 'n_heads': 4, 'd_ff': 170, 'n_loops': 2, 'seq_len': 128, 'train_batch_tokens': 1024, 'val_batch_tokens': 1024, 'val_steps': 0, 'val_loss_every': 0, 'max_steps': 1, 'max_wallclock_seconds': 0, 'avg_bytes_per_token': 2.442303811509075, 'tokenizer_path': '/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model', 'muon_lr': 0.02, 'adamw_lr': 0.0003, 'weight_decay': 0.1, 'warmup_steps': 20, 'cooldown_fraction': 0.3, 'qat_start_fraction': 0.6, 'grad_clip': 1.0, 'seed': 1337, 'device': 'cpu', 'compile_model': False, 'use_smear': True, 'artifact_path': '', 'stats_path': ''}
train_source=1 shard(s) val_source=1 shard(s) device=cpu
vocab_size=1024 source=env
token_dtype=uint16 source=env
avg_bytes_per_token=2.4423 source=env
bpb_mode=sentencepiece_exact
tokenizer_path=/private/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model
parameters=180,512
=== final_stats ===
steps=1
seconds=0.06
final_val_loss=6.9297
final_val_bpb=4.0297
compressed_model_size_bytes=160723
code_size_bytes=37388
total_artifact_bytes=198111
artifact_budget_ok=True
```

Interpretation:

- The exact SentencePiece path runs successfully in our trainer.
- The trainer can now combine:
  - official-format shard reading
  - exact tokenizer-aware `val_bpb`
  - sequential full-scan validation when requested

## Experiment 23. First Exact Sampled-Validation MPS Run on Official `sp1024`

Status:

- Passed

Purpose:

- Establish the first meaningful speed and score baseline on the new fast exact-bpb path.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_s100_v16 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=100 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_s100_v16.json .venv/bin/python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=7.0349 train_bpb=4.1259 val_loss=7.0171 val_bpb=4.1593 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=20 train_loss=5.8438 train_bpb=3.3846 val_loss=5.6938 val_bpb=3.3789 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=21.0s
step=40 train_loss=4.7811 train_bpb=2.7607 val_loss=4.7556 val_bpb=2.8222 muon_lr=1.378e-02 adamw_lr=2.067e-04 elapsed=42.7s
step=60 qat=enabled
step=60 train_loss=4.3375 train_bpb=2.5592 val_loss=4.3541 val_bpb=2.5808 muon_lr=3.719e-03 adamw_lr=5.578e-05 elapsed=63.8s
step=80 train_loss=4.5291 train_bpb=2.7209 val_loss=4.5187 val_bpb=2.6804 muon_lr=1.333e-03 adamw_lr=2.000e-05 elapsed=85.2s
=== final_stats ===
steps=100
seconds=106.44
final_val_loss=4.5915
final_val_bpb=2.7163
compressed_model_size_bytes=3492739
code_size_bytes=37388
total_artifact_bytes=3530127
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_s100_v16.json
```

Key parsed JSON result:

```json
{
  "best_val_bpb": 2.580798659621066,
  "final_val_bpb": 2.7162548714625445,
  "parameters": 4198016,
  "seconds": 106.44016912498046
}
```

Interpretation:

- This is the first local result that is both fast enough to iterate on and faithful to the official tokenizer's byte semantics.
- The learning curve is clearly real.
- The schedule is not good enough yet:
  - cooldown starts too early for such a short run
  - QAT likely arrived too soon for this regime
- Even with those flaws, this run materially improved on the older proxy-only neighborhood.

## Experiment 24. `64k` Batch Throughput Probe on the Exact Sampled MPS Path

Status:

- Passed

Purpose:

- Check whether a much larger effective batch improves tokens-per-second enough to justify longer official-tokenizer runs.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s20 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=32768 VAL_STEPS=4 VAL_LOSS_EVERY=10 MAX_STEPS=20 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=0.98 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s20.json .venv/bin/python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=7.0376 train_bpb=4.1401 val_loss=7.0170 val_bpb=4.1681 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=6.3463 train_bpb=3.7325 val_loss=6.2598 val_bpb=3.7078 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=32.5s
step=19 qat=enabled
=== final_stats ===
steps=20
seconds=132.00
final_val_loss=5.7401
final_val_bpb=3.4061
compressed_model_size_bytes=3513871
code_size_bytes=37388
total_artifact_bytes=3551259
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s20.json
```

Interpretation:

- The `64k` train batch is viable on MPS for this model.
- Throughput improved versus the `16k` batch loop:
  - `16k` path reached `step=20` in about `21.0s`
  - `64k` path reached `step=10` in about `32.5s`
- That is roughly `15.6k` tokens/s for the `16k` run versus about `20.2k` tokens/s for the `64k` probe.
- So larger batches are now part of the search space for longer runs.

## Experiment 25. Longer `64k` Batch Exact Sampled Run

Status:

- Passed

Purpose:

- Test whether the exact sampled-bpb curve keeps improving when we keep the faster `64k` batch, delay cooldown, and give the model a longer horizon.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s200 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=128 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=200 COOLDOWN_FRACTION=0.05 QAT_START_FRACTION=0.98 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s200.json .venv/bin/python train_gpt.py
```

Terminal output so far:

```text
step=0 train_loss=7.0376 train_bpb=4.1401 val_loss=7.0164 val_bpb=4.1588 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=50 train_loss=4.2241 train_bpb=2.4879 val_loss=4.1846 val_bpb=2.4833 muon_lr=1.865e-02 adamw_lr=2.798e-04 elapsed=156.9s
```

Final parsed stats after the run completed:

```json
{
  "run_id": "torch_sp1024_d512_l4_b64k_s200",
  "steps": 200,
  "seconds": 6587.595697958022,
  "best_val_loss": 3.3640987277030945,
  "best_val_bpb": 1.995522745133331,
  "final_val_loss": 3.3640987277030945,
  "final_val_bpb": 1.995522745133331,
  "compressed_model_size_bytes": 3529390,
  "total_artifact_bytes": 3566778,
  "artifact_budget_ok": true
}
```

Interpretation:

- This is the best exact sampled official-tokenizer score we have logged locally so far.
- The score improved dramatically from the earlier `2.5808` run even though the sequence length was still only `128`.
- This result is strong enough to confirm that the fast exact SentencePiece path is worth using, but it also sharpens the main reviewer criticism:
  - if `SEQ_LEN=128` can already reach `1.9955`
  - then fixing sequence length is now the highest-leverage next move

## Experiment 26. External Review Packet Draft

Status:

- Passed

Purpose:

- Create a single long-form document that summarizes the full project history for an outside reviewer.

Artifact created:

```text
DRAFT1.md
```

What it contains:

- complete commit history
- competition understanding and rule alignment
- full experiment timeline
- current best numbers
- major failures and dead ends
- active hypotheses and review questions

Interpretation:

- `STRATEGY.md` remains the raw lab notebook.
- `DRAFT1.md` is the cleaner handoff document for another engineer or researcher to review and critique.

## Experiment 27. Feedback Intake and Search Reset

Status:

- Passed

Purpose:

- Convert the external review in `FEEDBACK1.md` into explicit scorekeeping rules and a reordered search plan.

Key feedback adopted:

- `SEQ_LEN=128` is likely the biggest self-inflicted bottleneck
- sampled exact validation should be treated as roughly `+/- 0.05` bpb
- QAT should be disabled during local architecture search
- the official `0.005 nats` record bar needs to be tracked in `bpb`

Concrete changes to project guidance:

- recorded `0.005 nats = 0.007213475204444817 bpb`
- recorded the practical record-claim target as `1.217152224795555` bpb
- promoted the finished `1.995522745133331` run to the main scoreboard
- reset the next experiment priority away from tokenizer tinkering and toward sequence-length fixes first

Interpretation:

- The project is not bottlenecked on ideas right now.
- It is bottlenecked on running the current best official-tokenizer configuration in a non-broken long-context regime.

## Experiment 28. `SEQ_LEN=512` Feasibility and Pilot

Status:

- Passed

Purpose:

- Run the highest-priority reviewer suggestion directly: keep the current best `sp1024` model shape, disable QAT, lengthen the context window, and see whether `SEQ_LEN=512` is practical and helpful.

Smoke command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s1_seq512_smoke TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=512 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=65536 VAL_STEPS=16 VAL_LOSS_EVERY=0 MAX_STEPS=1 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s1_seq512_smoke.json uv run python train_gpt.py
```

Smoke terminal output:

```text
=== final_stats ===
steps=1
seconds=36.44
final_val_loss=6.7021
final_val_bpb=3.9695
compressed_model_size_bytes=3562545
code_size_bytes=37388
total_artifact_bytes=3599933
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s1_seq512_smoke.json
```

Pilot command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s20_seq512 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=512 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=65536 VAL_STEPS=16 VAL_LOSS_EVERY=10 MAX_STEPS=20 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s20_seq512.json uv run python train_gpt.py
```

Pilot terminal output:

```text
step=0 train_loss=7.0366 train_bpb=4.1504 val_loss=7.0158 val_bpb=4.1553 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=10 train_loss=6.3466 train_bpb=3.7293 val_loss=6.2837 val_bpb=3.7293 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=756.4s
=== final_stats ===
steps=20
seconds=1508.84
final_val_loss=5.7089
final_val_bpb=3.3877
compressed_model_size_bytes=3531857
code_size_bytes=37388
total_artifact_bytes=3569245
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s20_seq512.json
```

Interpretation:

- `SEQ_LEN=512` is feasible at the original `64k` token batch on this Mac.
- The `20`-step result improved slightly over the older `SEQ_LEN=128` `20`-step probe:
  - `3.3877` versus `3.4061`
- But the gain is only `0.0184` bpb, which is inside the local sampled-eval noise band.
- The more important lesson was throughput:
  - `SEQ_LEN=512` at `64k` batch is extremely slow locally
  - so `SEQ_LEN=1024` would need a smaller batch to be practical

## Experiment 29. `SEQ_LEN=1024` Memory Ceiling Probe

Status:

- Passed with fallback

Purpose:

- Determine the largest practical local batch for the `SEQ_LEN=1024` branch before spending more time on long-context training.

OOM command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b64k_s1_seq1024_smoke TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_TOKENS=65536 VAL_STEPS=16 VAL_LOSS_EVERY=0 MAX_STEPS=1 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b64k_s1_seq1024_smoke.json uv run python train_gpt.py
```

OOM terminal output:

```text
RuntimeError: MPS backend out of memory (MPS allocated: 27.89 GiB, other allocations: 2.07 GiB, max allowed: 30.19 GiB). Tried to allocate 256.00 MiB on private pool.
```

Fallback command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s1_seq1024_smoke TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=0 MAX_STEPS=1 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s1_seq1024_smoke.json uv run python train_gpt.py
```

Fallback terminal output:

```text
=== final_stats ===
steps=1
seconds=1.85
final_val_loss=6.7130
final_val_bpb=3.9736
compressed_model_size_bytes=3565073
code_size_bytes=37388
total_artifact_bytes=3602461
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s1_seq1024_smoke.json
```

Interpretation:

- `SEQ_LEN=1024` is not dead locally.
- It simply requires dropping the batch from `64k` to `32k`.
- That fallback is still fast enough to be useful for local search.

## Experiment 30. `SEQ_LEN=1024` Pilot and Token-Budget Match

Status:

- Passed

Purpose:

- Compare the `SEQ_LEN=1024` branch fairly by first running a short pilot, then matching the training-token budget of the `SEQ_LEN=512` `20`-step run.

Pilot command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s10_seq1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=10 MAX_STEPS=10 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s10_seq1024.json uv run python train_gpt.py
```

Pilot terminal output:

```text
step=0 train_loss=7.0330 train_bpb=4.0486 val_loss=7.0155 val_bpb=4.1527 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
=== final_stats ===
steps=10
seconds=28.11
final_val_loss=6.3621
final_val_bpb=3.7702
compressed_model_size_bytes=3567268
code_size_bytes=37388
total_artifact_bytes=3604656
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s10_seq1024.json
```

Token-budget-matched command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s40_seq1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s40_seq1024.json uv run python train_gpt.py
```

Token-budget-matched terminal output:

```text
step=0 train_loss=7.0330 train_bpb=4.0486 val_loss=7.0155 val_bpb=4.1527 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=20 train_loss=5.8047 train_bpb=3.4195 val_loss=5.7390 val_bpb=3.4009 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=43.6s
=== final_stats ===
steps=40
seconds=90.60
final_val_loss=5.1800
final_val_bpb=3.0763
compressed_model_size_bytes=3522792
code_size_bytes=37388
total_artifact_bytes=3560180
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s40_seq1024.json
```

Interpretation:

- This is where the feedback started looking decisively correct.
- At matched training-token budget, `SEQ_LEN=1024` beat the `SEQ_LEN=512` pilot by a large margin:
  - `3.0763` versus `3.3877`
- Long context was not the dead end; unfair step-count comparisons were.

## Experiment 31. `SEQ_LEN=1024` `200`-Step Long-Context Run

Status:

- Passed

Purpose:

- Push the now-validated `SEQ_LEN=1024`, `32k` batch regime to a proper longer run with no QAT and the revised cooldown.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s200_seq1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=200 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s200_seq1024.json uv run python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=7.0330 train_bpb=4.0486 val_loss=7.0155 val_bpb=4.1527 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=50 train_loss=4.5096 train_bpb=2.6665 val_loss=4.4793 val_bpb=2.6544 muon_lr=1.774e-02 adamw_lr=2.660e-04 elapsed=101.5s
step=100 train_loss=4.1133 train_bpb=2.4946 val_loss=3.9488 val_bpb=2.3451 muon_lr=7.809e-03 adamw_lr=1.171e-04 elapsed=201.1s
step=150 train_loss=3.8094 train_bpb=2.2180 val_loss=3.7763 val_bpb=2.2305 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=300.0s
=== final_stats ===
steps=200
seconds=400.57
final_val_loss=3.7196
final_val_bpb=2.2033
compressed_model_size_bytes=3544560
code_size_bytes=37388
total_artifact_bytes=3581948
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s200_seq1024.json
```

Interpretation:

- The long-context branch is real and stable.
- It did not beat the older `1.9955` sampled best yet, but it also used only half the training-token budget of that run.
- The right comparison was therefore not `200` steps versus `200` steps; it was matched total tokens.

## Experiment 32. `SEQ_LEN=1024` `400`-Step Matched-Token Run

Status:

- Passed

Purpose:

- Match the total training-token budget of the older `SEQ_LEN=128`, `64k`, `200`-step run and see whether long context closes the gap when compared fairly.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s400_seq1024 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=100 MAX_STEPS=400 COOLDOWN_FRACTION=0.25 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s400_seq1024.json uv run python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=7.0330 train_bpb=4.0486 val_loss=7.0155 val_bpb=4.1527 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=100 train_loss=4.1389 train_bpb=2.5102 val_loss=3.9680 val_bpb=2.3514 muon_lr=1.661e-02 adamw_lr=2.492e-04 elapsed=177.9s
step=200 train_loss=3.5787 train_bpb=2.0785 val_loss=3.5770 val_bpb=2.1243 muon_lr=7.095e-03 adamw_lr=1.064e-04 elapsed=360.5s
step=300 train_loss=3.4236 train_bpb=1.9999 val_loss=3.4518 val_bpb=2.0388 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=544.6s
=== final_stats ===
steps=400
seconds=727.44
final_val_loss=3.4013
final_val_bpb=2.0147
compressed_model_size_bytes=3545750
code_size_bytes=37388
total_artifact_bytes=3583138
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s400_seq1024.json
```

Interpretation:

- This is the key result from the feedback-driven reset.
- At matched total training tokens, the best long-context branch is now only `0.0192` bpb worse than the old sampled best `1.9955`.
- That difference is inside the local sampled-eval noise band, so we should treat the two branches as roughly tied for now.
- The long-context branch is therefore not a regression.
- It is the better search base going forward because:
  - it obeys the reviewer's core argument about context
  - it runs much more predictably on the local machine
  - it gives us a cleaner regime for schedule tuning next

## Experiment 33. `SEQ_LEN=1024` Schedule Tuning: `COOLDOWN_FRACTION=0.30`

Status:

- Passed

Purpose:

- Test the next feedback-suggested cooldown point while holding the validated long-context regime fixed.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s400_seq1024_cd30 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=100 MAX_STEPS=400 COOLDOWN_FRACTION=0.30 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s400_seq1024_cd30.json uv run python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=7.0330 train_bpb=4.0486 val_loss=7.0155 val_bpb=4.1527 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
step=100 train_loss=4.1367 train_bpb=2.5088 val_loss=3.9693 val_bpb=2.3522 muon_lr=1.611e-02 adamw_lr=2.417e-04 elapsed=176.2s
step=200 train_loss=3.5799 train_bpb=2.0792 val_loss=3.5779 val_bpb=2.1249 muon_lr=5.887e-03 adamw_lr=8.831e-05 elapsed=357.1s
step=300 train_loss=3.4364 train_bpb=2.0074 val_loss=3.4660 val_bpb=2.0472 muon_lr=1.667e-03 adamw_lr=2.500e-05 elapsed=545.0s
=== final_stats ===
steps=400
seconds=731.03
final_val_loss=3.4199
final_val_bpb=2.0258
compressed_model_size_bytes=3541387
code_size_bytes=37388
total_artifact_bytes=3578775
artifact_budget_ok=True
stats_path=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s400_seq1024_cd30.json
```

Interpretation:

- `COOLDOWN_FRACTION=0.30` is slightly worse than `0.25` in this regime:
  - `2.0258` versus `2.0147`
- The difference is small, but the direction is consistent at intermediate checkpoints too.
- So the feedback-driven schedule sweep has already ruled out one branch.
- The next fair comparison is `COOLDOWN_FRACTION=0.20`.

## Experiment 34. `SEQ_LEN=1024` Schedule Tuning: `COOLDOWN_FRACTION=0.20`

Status:

- Passed

Purpose:

- Complete the reviewer-suggested local cooldown sweep and see whether a slightly earlier cooldown beats the `0.25` baseline.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=torch_sp1024_d512_l4_b32k_s400_seq1024_cd20 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=100 MAX_STEPS=400 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/torch_sp1024_d512_l4_b32k_s400_seq1024_cd20.json uv run python train_gpt.py
```

Terminal output summary:

```text
final_val_bpb=2.0050
compressed_model_size_bytes=3541574
total_artifact_bytes=3581567
artifact_budget_ok=True
```

Interpretation:

- `COOLDOWN_FRACTION=0.20` beat both `0.25` and `0.30` in the current long-context regime.
- The gain over `0.25` is small:
  - `2.0050` versus `2.0147`
- But this is now the best long-context result we have, and it leaves the long-context branch effectively tied with the old `1.9955` sampled best.

## Experiment 35. Add Exact ByteLevel BPE `val_bpb`, Tied Embeddings, and Untied-Stack Option

Status:

- Passed

Purpose:

- Re-open the larger-tokenizer path properly after the long-context reset.
- Support exact byte accounting for our existing ByteLevel BPE tokenizers.
- Add `TIED_EMBEDDINGS=1` so large-vocab experiments do not waste half their parameters on a duplicate output matrix.
- Add `SHARE_BLOCKS=0` so we can test a baseline-like untied stack instead of only the shared-loop architecture.

Key code ideas:

```python
class ByteLevelBPBHelper:
    def __init__(self, byte_length_lut) -> None:
        self.byte_length_lut = byte_length_lut
```

```python
if self.tied_embeddings:
    self.head.linear.weight = self.embed.weight
```

```python
if self.share_blocks:
    for loop_idx in range(self.n_loops):
        x = x + loop_emb[loop_idx].view(1, 1, -1)
        x = self.block(x, cos, sin)
else:
    for block in self.blocks:
        x = block(x, cos, sin)
```

Interpretation:

- We now have exact local `val_bpb` for our `16k` and `24k` ByteLevel BPE tokenizers instead of only average-bytes proxies.
- We can now test:
  - large vocab + tied embeddings
  - large vocab + exact byte accounting
  - shared loops versus a standard untied stack

## Experiment 36. Exact ByteLevel Tokenizer Re-Entry: `24k` and `16k`

Status:

- Passed

Purpose:

- Re-test the larger tokenizer idea after fixing context length and adding exact byte accounting.

`24k` exact `40`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_s40.json uv run python train_gpt.py
```

`24k` terminal output:

```text
step=20 train_loss=8.4139 train_bpb=2.8644 val_loss=8.2890 val_bpb=2.8128 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=29.6s
=== final_stats ===
steps=40
seconds=60.43
final_val_bpb=2.5938
total_artifact_bytes=23708605
artifact_budget_ok=False
```

`24k` exact `400`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_s400 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=100 MAX_STEPS=400 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_s400.json uv run python train_gpt.py
```

`24k` terminal output summary:

```text
step=100 ... val_bpb=2.2806
step=200 ... val_bpb=2.1238
step=300 ... val_bpb=2.1126
=== final_stats ===
steps=400
seconds=561.85
final_val_bpb=2.0819
total_artifact_bytes=23511822
artifact_budget_ok=False
```

`16k` exact `40`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel16k_d512_tied_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_16k_sample DATA_PATH=./data/tokens/fineweb_16k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_16k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel16k/bytelevel16k_d512_tied_s40.json uv run python train_gpt.py
```

`16k` terminal output:

```text
step=20 train_loss=8.1166 train_bpb=2.9432 val_loss=8.0460 val_bpb=2.8624 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=26.6s
=== final_stats ===
steps=40
seconds=53.19
final_val_bpb=2.6767
total_artifact_bytes=16564231
artifact_budget_ok=False
```

Interpretation:

- The larger-tokenizer idea was not a mirage, but it was also not the instant path to `1.5`.
- `24k` is clearly better than `16k` in this regime:
  - `2.0819` versus `2.6767`
- But both large-vocab `d_model=512` tied-embedding models are still over the artifact cap, and the `24k` curve flattened above `2.0`.

## Experiment 37. Baseline-Family Stack Check

Status:

- Passed

Purpose:

- Test a more official-looking untied `9`-layer stack with tied embeddings on `sp1024`.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=sp1024_stack9_s100 TOKENIZER_PATH=/tmp/parameter-golf-official.BbMmsz/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VAL_DATA_PATH=/tmp/parameter-golf-official.BbMmsz/data/datasets/fineweb10B_sp1024 VOCAB_SIZE=1024 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=2.442303811509075 D_MODEL=512 N_HEADS=8 D_FF=1024 N_LOOPS=9 SHARE_BLOCKS=0 TIED_EMBEDDINGS=1 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=100 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 DEVICE=mps STATS_PATH=runs/official_torch_sp1024/sp1024_stack9_s100.json uv run python train_gpt.py
```

Terminal output:

```text
step=50 train_loss=4.6067 train_bpb=2.8051 val_loss=4.6022 val_bpb=2.7242 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=99.0s
=== final_stats ===
steps=100
seconds=200.91
final_val_bpb=2.5413
total_artifact_bytes=20390813
artifact_budget_ok=False
```

Interpretation:

- The baseline-family comparison was useful, but it was not the win.
- This `9`-layer stack underperformed the best long-context shared-loop branch in our local regime.
- That means the search should stay on the shared-loop family for now unless we revisit the stack with a very different schedule.

## Experiment 38. Deeper Shared Loops on `24k`

Status:

- Passed

Purpose:

- Re-test loop depth in the corrected long-context regime, since the old short-context sweeps were likely misleading.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_l8_s100 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=8 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=100 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_l8_s100.json uv run python train_gpt.py
```

Terminal output:

```text
step=50 train_loss=7.2609 train_bpb=2.4711 val_loss=7.2797 val_bpb=2.4704 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=510.5s
=== final_stats ===
steps=100
seconds=852.14
final_val_bpb=2.3682
total_artifact_bytes=23373548
artifact_budget_ok=False
```

Interpretation:

- More loops were not the answer here.
- The `8`-loop run was much slower and still worse than the `4`-loop `24k` branch at comparable training budget.
- So the old short-context "extra loops are not helping" conclusion still seems directionally true in this family.

## Experiment 39. `24k` Larger Effective Batch Probe

Status:

- Passed with OOM fallback

Purpose:

- Check whether the `24k` exact branch was being unfairly undertrained because it used only `16k` train tokens per step.

`32k` smoke command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_b32k_smoke TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=4 VAL_LOSS_EVERY=0 MAX_STEPS=1 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_b32k_smoke.json uv run python train_gpt.py
```

Smoke terminal output:

```text
=== final_stats ===
steps=1
seconds=19.84
final_val_bpb=3.3783
```

`32k` `100`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_b32k_s100 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_TOKENS=32768 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=100 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_b32k_s100.json uv run python train_gpt.py
```

Terminal output:

```text
step=0 train_loss=10.2199 train_bpb=3.3598 val_loss=10.1983 val_bpb=3.4764 muon_lr=1.000e-03 adamw_lr=1.500e-05 elapsed=0.0s
RuntimeError: MPS backend out of memory (MPS allocated: 24.21 GiB, other allocations: 4.05 GiB, max allowed: 30.19 GiB). Tried to allocate 3.00 GiB on private pool.
```

Interpretation:

- `24k` at `32k` train tokens per step does fit for a smoke step but fails during sustained training.
- So the current local hardware limit for this branch is still effectively the `16k` train-token regime unless we add gradient accumulation later.

## Experiment 40. Gradient Accumulation for the `24k` Branch

Status:

- Passed

Purpose:

- Recover a larger effective train batch on the promising `24k` exact branch without triggering MPS OOMs.

Key code idea:

```python
train_microbatch_tokens = cfg.train_microbatch_tokens or cfg.train_batch_tokens
train_micro_bsz = max(1, train_microbatch_tokens // cfg.seq_len)
grad_accum_steps = max(1, math.ceil(train_bsz / train_micro_bsz))
```

```python
for _ in range(grad_accum_steps):
    batch = train_batcher.next_batch(train_micro_bsz, device)
    ...
    scaled_loss = loss / grad_accum_steps
    scaled_loss.backward()
```

Smoke command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_eff32k_accum_smoke TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 TRAIN_MICROBATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=4 VAL_LOSS_EVERY=0 MAX_STEPS=1 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_eff32k_accum_smoke.json uv run python train_gpt.py
```

Smoke terminal output:

```text
train_batch_size_tokens=32768 train_microbatch_size_tokens=16384 grad_accum_steps=2
=== final_stats ===
steps=1
seconds=3.12
final_val_bpb=3.4068
```

Pilot command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_eff32k_accum_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=32768 TRAIN_MICROBATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_eff32k_accum_s40.json uv run python train_gpt.py
```

Pilot terminal output:

```text
train_batch_size_tokens=32768 train_microbatch_size_tokens=16384 grad_accum_steps=2
step=20 train_loss=8.3142 train_bpb=2.8107 val_loss=8.2653 val_bpb=2.8048 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=52.7s
=== final_stats ===
steps=40
seconds=106.50
final_val_bpb=2.5815
total_artifact_bytes=23756385
artifact_budget_ok=False
```

Interpretation:

- Gradient accumulation works and gives us a stable way to exceed the direct-memory limit of the `24k` branch.
- The first accumulated `32k` effective-batch result is only a small improvement over the old direct `16k` batch result:
  - `2.5815` versus `2.5938`
- But it is an improvement, and it unlocks the next clean experiment:
  - effective `64k` train tokens per step using four `16k` microbatches

## Experiment 41. Effective `64k` Batch on `24k`

Status:

- Passed

Purpose:

- Test whether the promising `24k` branch was mostly limited by effective batch size.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_eff64k_accum_s100 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=65536 TRAIN_MICROBATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=50 MAX_STEPS=100 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_eff64k_accum_s100.json uv run python train_gpt.py
```

Terminal output:

```text
step=50 train_loss=7.1961 train_bpb=2.4480 val_loss=7.1237 val_bpb=2.4174 muon_lr=1.100e-02 adamw_lr=1.650e-04 elapsed=242.9s
=== final_stats ===
steps=100
seconds=502.74
final_val_bpb=2.3007
total_artifact_bytes=23578062
artifact_budget_ok=False
```

Interpretation:

- Bigger effective batch by itself was not enough.
- At matched training-token budget, this was still worse than the best older `24k` run.
- That pointed to learning-rate mismatch rather than batch size as the dominant issue.

## Experiment 42. Learning-Rate Retune on `24k`

Status:

- Passed

Purpose:

- Check whether the underwhelming accumulated-batch result was caused by using the old learning rates in a new effective-batch regime.

`2x` LR direct `24k` `40`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_lr2x_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_lr2x_s40.json uv run python train_gpt.py
```

`2x` LR terminal output:

```text
step=20 train_loss=7.8087 train_bpb=2.6583 val_loss=7.7303 val_bpb=2.6233 muon_lr=4.000e-02 adamw_lr=6.000e-04 elapsed=29.4s
=== final_stats ===
steps=40
seconds=58.52
final_val_bpb=2.4865
```

`3x` LR direct `24k` `40`-step command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_lr3x_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.06 ADAMW_LR=0.0009 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_lr3x_s40.json uv run python train_gpt.py
```

`3x` LR terminal output:

```text
step=20 train_loss=7.6637 train_bpb=2.6089 val_loss=7.6019 val_bpb=2.5797 muon_lr=6.000e-02 adamw_lr=9.000e-04 elapsed=29.6s
=== final_stats ===
steps=40
seconds=58.65
final_val_bpb=2.4674
```

Interpretation:

- Learning-rate retuning mattered more than the raw batch increase.
- `3x` was slightly better than the old baseline but slightly worse than `2x` at the end of the short pilot.
- So `2x` became the promoted setting for the best current `24k` branch.

## Experiment 43. Promoted `24k` `2x` LR Run

Status:

- Passed

Purpose:

- Extend the best short `24k` LR-retuned branch into a proper long run.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_lr2x_s400 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=100 MAX_STEPS=400 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_lr2x_s400.json uv run python train_gpt.py
```

Terminal output summary:

```text
step=100 ... val_bpb=2.2428
step=200 ... val_bpb=2.0878
step=300 ... val_bpb=2.0704
=== final_stats ===
steps=400
seconds=488.03
final_val_bpb=2.0389
total_artifact_bytes=23422947
artifact_budget_ok=False
```

Interpretation:

- This is now the best exact `24k` result we have.
- The LR retune improved the old `24k` `400`-step branch from `2.0819` to `2.0389`.
- That is a real improvement, but still nowhere near the `1.5` local target.

## Experiment 44. Wider `24k` Check After LR Retuning

Status:

- Passed

Purpose:

- Check whether width still helps once the `24k` branch has a much better learning rate.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_lr2x_s40 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=20 MAX_STEPS=40 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_lr2x_s40.json uv run python train_gpt.py
```

Terminal output:

```text
step=20 train_loss=7.7260 train_bpb=2.6302 val_loss=7.6901 val_bpb=2.6096 muon_lr=4.000e-02 adamw_lr=6.000e-04 elapsed=37.2s
=== final_stats ===
steps=40
seconds=75.08
final_val_bpb=2.4844
```

Interpretation:

- Width did not buy a meaningful gain here.
- `d_model=640` was only trivially better than `d_model=512` at the same short-horizon setting.
- That means the next higher-value move is more optimization on the `d_model=512` branch, not a width jump.

## Experiment 45. Extended `24k` `2x` LR Run To 800 Steps

Status:

- Passed

Purpose:

- Find out whether the promoted `24k` `2x` LR branch was still descending materially after the `400`-step milestone.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_lr2x_s800 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=200 MAX_STEPS=800 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_lr2x_s800.json uv run python train_gpt.py
```

Terminal output summary:

```text
step=200 ... val_bpb=2.0980
step=400 ... val_bpb=1.9514
step=600 ... val_bpb=1.9351
=== final_stats ===
steps=800
seconds=936.27
final_val_bpb=1.9059
total_artifact_bytes=22746326
artifact_budget_ok=False
```

Interpretation:

- This is now the best overall exact local score in the repo.
- Extending the same branch from `400` to `800` steps improved `final_val_bpb` from `2.0389` to `1.9059`.
- The improvement from the extra horizon was much larger than the recent width or accumulation experiments.
- The branch is still badly over the artifact cap, but it is now much closer to the local `1.5` target than any official-tokenizer branch.

## Experiment 46. Add Baseline-Aligned GQA and Logit Softcap Knobs

Status:

- Passed

Purpose:

- Add the next most plausible official-baseline-alignment levers to the local trainer before the current `24k` branch saturates:
  - grouped-query attention via `NUM_KV_HEADS`
  - per-head query gain via `QK_GAIN_INIT`
  - output `LOGIT_SOFTCAP`

Code snippet:

```python
self.q_proj = QATLinear(d_model, d_model, bias=False)
self.k_proj = QATLinear(d_model, kv_dim, bias=False)
self.v_proj = QATLinear(d_model, kv_dim, bias=False)
self.q_gain = nn.Parameter(torch.full((n_heads,), qk_gain_init))
...
if self.kv_repeat > 1:
    k = k.repeat_interleave(self.kv_repeat, dim=1)
    v = v.repeat_interleave(self.kv_repeat, dim=1)
...
if self.logit_softcap > 0.0:
    logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
```

Verification commands:

```bash
python3 -m py_compile train_gpt.py
RUN_ID=gqa_softcap_smoke DEVICE=cpu MAX_STEPS=1 VAL_LOSS_EVERY=1 VAL_STEPS=1 D_MODEL=64 N_HEADS=8 NUM_KV_HEADS=4 D_FF=128 N_LOOPS=2 SEQ_LEN=32 TRAIN_BATCH_TOKENS=256 VAL_BATCH_TOKENS=256 VOCAB_SIZE=512 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=3.5 TIED_EMBEDDINGS=1 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 uv run python train_gpt.py
```

Terminal output:

```text
config: {'run_id': 'gqa_softcap_smoke', ..., 'n_heads': 8, 'num_kv_heads': 4, ..., 'qk_gain_init': 1.5, 'logit_softcap': 30.0, ...}
parameters=70,040
step=0 train_loss=6.2254 train_bpb=2.5661 val_loss=6.2452 val_bpb=2.5743 muon_lr=2.000e-02 adamw_lr=3.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=0.08
final_val_bpb=2.5757
artifact_budget_ok=True
```

Interpretation:

- The new baseline-aligned knobs compile and execute correctly.
- Defaults remain backwards-compatible, so older experiments still reproduce unless the new flags are set.
- The currently running `bytelevel24k_d512_tied_lr2x_s1600` experiment was launched before this patch, so its result should be compared only against the pre-GQA/softcap code snapshot.
- If the long `24k` branch stalls, the next architecture ablation should test `NUM_KV_HEADS=4`, `QK_GAIN_INIT=1.5`, and `LOGIT_SOFTCAP=30`.

## Experiment 47. Extend The Promoted `24k` Branch To 1600 Steps

Status:

- Passed

Purpose:

- Test whether the now-promoted `24k` `2x` LR branch still had a meaningful amount of headroom once the total run length and cooldown horizon were doubled.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_tied_lr2x_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_tied_lr2x_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d512_tied_lr2x_s1600.log
```

Terminal output:

```text
step=400 train_loss=5.5973 train_bpb=1.9005 val_loss=5.7808 val_bpb=1.9617 muon_lr=3.251e-02 adamw_lr=4.876e-04 elapsed=459.5s
step=800 train_loss=5.3514 train_bpb=1.7741 val_loss=5.4348 val_bpb=1.8308 muon_lr=1.542e-02 adamw_lr=2.314e-04 elapsed=915.9s
step=1200 train_loss=5.0116 train_bpb=1.7362 val_loss=5.3021 val_bpb=1.8121 muon_lr=4.357e-03 adamw_lr=6.535e-05 elapsed=1377.5s
=== final_stats ===
steps=1600
seconds=1838.28
final_val_bpb=1.7854
total_artifact_bytes=22405986
artifact_budget_ok=False
```

Interpretation:

- This is the new best local exact result in the repo.
- Relative to the old `800`-step `24k` best (`1.9059`), the longer-horizon run improved by `0.1206` bpb.
- The extra horizon and later cooldown mattered much more than recent width, loop-depth, or accumulation changes.
- We are still `0.2854` bpb away from the local `1.5` target, so longer training alone probably is not sufficient.

## Experiment 48. Add GQA, Query Gain, and Logit Softcap To The Best `1600`-Step Branch

Status:

- Passed

Purpose:

- Test whether the new baseline-aligned attention knobs improve the strongest `24k` long-horizon branch when everything else stays fixed.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_gqa_softcap_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_s1600.log
```

Terminal output:

```text
parameters=15,470,216
step=400 train_loss=5.4697 train_bpb=1.8571 val_loss=5.6657 val_bpb=1.9226 muon_lr=3.251e-02 adamw_lr=4.876e-04 elapsed=482.8s
step=800 train_loss=5.2379 train_bpb=1.7365 val_loss=5.3298 val_bpb=1.7954 muon_lr=1.542e-02 adamw_lr=2.314e-04 elapsed=964.4s
step=1200 train_loss=4.8926 train_bpb=1.6949 val_loss=5.1954 val_bpb=1.7756 muon_lr=4.357e-03 adamw_lr=6.535e-05 elapsed=1443.8s
=== final_stats ===
steps=1600
seconds=1922.98
final_val_bpb=1.7484
total_artifact_bytes=22076698
artifact_budget_ok=False
```

Interpretation:

- This is the new best local exact result in the repo.
- Relative to the plain `1600`-step branch (`1.7854`), the GQA/softcap branch improved by `0.0369` bpb.
- The variant also reduced parameters from `15,732,352` to `15,470,216` and trimmed artifact size by `329,288` bytes.
- GQA plus query gain plus logit softcap is now a confirmed useful direction, not just a theoretical baseline-alignment idea.

## Experiment 49. Extend The Improved GQA/Softcap Branch To 3200 Steps

Status:

- Passed

Purpose:

- Find out whether the improved `24k` GQA/softcap branch still had substantial optimization headroom beyond `1600` steps.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=800 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_s3200.log
```

Terminal output:

```text
step=800 train_loss=5.2046 train_bpb=1.7254 val_loss=5.3292 val_bpb=1.8085 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=951.1s
step=1600 train_loss=4.6848 train_bpb=1.5847 val_loss=4.9467 val_bpb=1.6664 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=1901.8s
step=2400 train_loss=4.5108 train_bpb=1.4575 val_loss=4.8383 val_bpb=1.6536 muon_lr=4.351e-03 adamw_lr=6.527e-05 elapsed=2853.4s
=== final_stats ===
steps=3200
seconds=3803.94
final_val_bpb=1.6238
total_artifact_bytes=21566256
artifact_budget_ok=False
```

Interpretation:

- This is the new best completed local exact result in the repo.
- Relative to the `1600`-step GQA/softcap branch (`1.7484`), the `3200`-step run improved by `0.1246` bpb.
- The extra horizon still helped a lot, but the last segment only improved `1.6664` to `1.6238`, so horizon alone is starting to flatten.
- We are now only `0.1238` bpb away from the local `1.5` target.

## Experiment 50. Prepare A `32k` Tokenizer Branch

Status:

- Passed

Purpose:

- Prepare the next most plausible denominator-improvement lever now that the `24k` branch is in the low `1.6`s but not yet at `1.5`.

Commands:

```bash
uv run python train_tokenizer.py --input-file data/raw/fineweb_16k_sample/train.jsonl --vocab-size 32768 --output-dir data/tokenizers --prefix fineweb_32k_sample
uv run python prepare_tokens.py --train-input-file data/raw/fineweb_16k_sample/train.jsonl --val-input-file data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_32k_sample --output-dir ./data/tokens/fineweb_32k_sample
```

Terminal output summary:

```text
source=local-files
vocab_path=data/tokenizers/fineweb_32k_sample-vocab.json
merges_path=data/tokenizers/fineweb_32k_sample-merges.txt
raw_size_bytes=811175
compressed_size_bytes=340918
train avg_bytes_per_token=4.4202857706695236
val avg_bytes_per_token=4.351551413669658
```

Interpretation:

- The `32k` tokenizer is ready for model training.
- Relative to the `24k` branch, this increases bytes per token from about `4.25` to about `4.42` on train data.
- That is exactly the kind of denominator gain that could matter once the model-side branch is already near `1.6`.

## Experiment 51. Prepare `48k` And `64k` Tokenizer Contingencies

Status:

- Passed

Purpose:

- Build a tokenizer ladder behind the active `32k` run so we can keep climbing the denominator without losing a setup cycle if `32k` still misses `1.5`.

Commands:

```bash
uv run python train_tokenizer.py --input-file data/raw/fineweb_16k_sample/train.jsonl --vocab-size 49152 --output-dir data/tokenizers --prefix fineweb_48k_sample
uv run python prepare_tokens.py --train-input-file data/raw/fineweb_16k_sample/train.jsonl --val-input-file data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_48k_sample --output-dir ./data/tokens/fineweb_48k_sample
uv run python train_tokenizer.py --input-file data/raw/fineweb_16k_sample/train.jsonl --vocab-size 65535 --output-dir data/tokenizers --prefix fineweb_64k_sample
uv run python prepare_tokens.py --train-input-file data/raw/fineweb_16k_sample/train.jsonl --val-input-file data/raw/fineweb_16k_sample/val.jsonl --tokenizer-prefix ./data/tokenizers/fineweb_64k_sample --output-dir ./data/tokens/fineweb_64k_sample
```

Terminal output summary:

```text
fineweb_48k_sample train avg_bytes_per_token=4.550211732381878
fineweb_48k_sample val avg_bytes_per_token=4.466813263206576
fineweb_64k_sample train avg_bytes_per_token=4.62566836285424
fineweb_64k_sample val avg_bytes_per_token=4.534385498540016
```

Interpretation:

- The tokenizer ladder is now `24k -> 32k -> 48k -> 64k`.
- Each rung materially increases bytes per token, so if the model-side branch is the same quality in nats, the bpb denominator should keep improving.
- We still need actual training runs to know whether the larger vocabularies hurt optimization enough to offset that benefit.

## Experiment 52. Mid-Run Checkpoint For The `32k` Branch

Status:

- In progress

Purpose:

- Check whether the live `32k` tokenizer branch is still on track to beat the `24k` branch, or whether the denominator gain is getting offset by weaker optimization.

Command:

```bash
tail -n 80 runs/bytelevel32k/bytelevel32k_d512_gqa_softcap_s3200.log
```

Terminal output:

```text
step=0 train_loss=10.4906 train_bpb=3.3573 val_loss=10.4514 val_bpb=3.4403 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=800 train_loss=5.0753 train_bpb=1.6506 val_loss=5.4205 val_bpb=1.7785 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1279.1s
step=1600 train_loss=4.7604 train_bpb=1.5083 val_loss=5.1868 val_bpb=1.6975 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=2624.8s
```

Interpretation:

- The `32k` branch is alive and making real progress.
- It looked better than `24k` at `step=800`, but by `step=1600` it is now `0.0311` bpb worse than the `24k` branch at the same checkpoint (`1.6975` versus `1.6664`).
- That means `32k` is no longer the obvious winning tokenizer rung.
- We should still let it finish, because the larger denominator might still help late in training, but `48k` is now the next most plausible escalation if `32k` stalls in the mid-`1.6`s.

## Experiment 54. Promote `32k` To `Step=2400`, Then Cut It Early

Status:

- Passed

Purpose:

- Wait long enough to tell whether the `32k` tokenizer branch had any realistic chance of reaching `1.5`, then stop it early if it was clearly not the best use of more MPS time.

Command:

```bash
tail -n 80 runs/bytelevel32k/bytelevel32k_d512_gqa_softcap_s3200.log
```

Terminal output:

```text
step=800 train_loss=5.0753 train_bpb=1.6506 val_loss=5.4205 val_bpb=1.7785 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1279.1s
step=1600 train_loss=4.7604 train_bpb=1.5083 val_loss=5.1868 val_bpb=1.6975 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=2624.8s
step=2400 train_loss=4.5696 train_bpb=1.4266 val_loss=5.0579 val_bpb=1.6675 muon_lr=4.351e-03 adamw_lr=6.527e-05 elapsed=4055.2s
```

Stop command:

```bash
^C
```

Interpretation:

- The `32k` branch kept improving, but not fast enough.
- At `step=2400`, it was still `0.0139` bpb worse than the `24k` branch at the same checkpoint (`1.6675` versus `1.6536`).
- More importantly, it would have needed another `0.1675` bpb drop in the final `800` steps to hit `1.5`, which was not realistic given the observed curve.
- Stopping it early and reallocating the device to `48k` was the highest-value move.

## Experiment 55. Launch The `48k` Tokenizer Branch

Status:

- Rejected in its original full-width form

Purpose:

- Promote the next tokenizer rung immediately after the `32k` branch failed to show a plausible path to `1.5`.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d512_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=800 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d512_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d512_gqa_softcap_s3200.log
```

Terminal output:

```text
vocab_size=49152 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_48k_sample/val
avg_bytes_per_token=4.4668 source=metadata:/Users/aryan/Desktop/golf/data/tokens/fineweb_48k_sample/val
parameters=28,053,128
step=0 train_loss=10.8960 train_bpb=3.4711 val_loss=10.8959 val_bpb=3.5214 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
```

Interpretation:

- The `48k` branch starts from a significantly larger denominator than `24k` or `32k`.
- The parameter count jumped a lot because the tied embedding table grew with the vocabulary.
- This exact `d512` form did start, but it was not healthy enough to keep.

## Experiment 56. Reject Full-Width `48k` On MPS For Memory Reasons

Status:

- Passed

Purpose:

- Verify whether the original `48k d512` branch was merely slow or actually a bad MPS memory configuration.

Commands:

```bash
top -pid 1972 -l 1 -stats pid,command,cpu,mem,time,threads,state
vm_stat | head -n 20
```

Terminal output:

```text
PID   COMMAND    %CPU MEM TIME     #TH  STATE
1972  python3.13 0.0  27G 03:11.47 13/1 running

PhysMem: 23G used (9122M wired, 3518M compressor), 59M unused.
VM: 185T vsize, 5268M framework vsize, 179957826(0) swapins, 194681707(0) swapouts.
```

Interpretation:

- The full-width `48k d512` branch was not just "slow".
- It was thrashing memory badly enough to be a poor search branch on this machine.
- That meant the next correct move was to keep the `48k` tokenizer rung but resize the model and lower peak memory.

## Experiment 57. Probe A Resized `48k d384` Branch

Status:

- Passed

Purpose:

- Keep the `48k` tokenizer denominator gain while moving back toward the parameter scale that had already worked for `32k d512`.

Parameter-count reference:

```text
24576 512 1365 15470216
32768 512 1365 19664520
49152 384 1024 20499560
49152 320 853 16857368
```

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d384_gqa_softcap_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_s3200.log
```

Terminal output before stop:

```text
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4639 val_loss=10.8440 val_bpb=3.5047 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
^C
```

Interpretation:

- `48k d384` was much closer to the successful `32k d512` parameter scale.
- It could start and reach `step=0` cleanly.
- But the direct full-batch form was still not the safest MPS configuration, so the right next move was gradient accumulation rather than abandoning the branch.

## Experiment 58. Validate Microbatching For `48k d384`

Status:

- Passed

Purpose:

- Confirm that `TRAIN_MICROBATCH_TOKENS=8192` fixes the peak-memory behavior while keeping the effective batch at `16384`.

Command:

```bash
RUN_ID=bytelevel48k_d384_gqa_softcap_accum_smoke TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 TRAIN_MICROBATCH_TOKENS=8192 VAL_BATCH_TOKENS=8192 VAL_STEPS=1 VAL_LOSS_EVERY=1 MAX_STEPS=1 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_smoke.json uv run python train_gpt.py
```

Terminal output:

```text
train_batch_size_tokens=16384 train_microbatch_size_tokens=8192 grad_accum_steps=2
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4118 val_loss=10.4851 val_bpb=3.5006 muon_lr=4.000e-02 adamw_lr=6.000e-04 elapsed=0.0s
=== final_stats ===
steps=1
seconds=24.29
final_val_bpb=3.3818
```

Interpretation:

- Accumulation worked.
- This is the first `48k` configuration that preserves the effective batch and finishes cleanly on this Mac.
- That made it the correct live frontier.

## Experiment 59. Launch The Real `48k d384` Accumulated Run

Status:

- In progress

Purpose:

- Continue denominator scaling with the first `48k` configuration that is actually healthy on local MPS.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel48k_d384_gqa_softcap_accum_s3200 TOKENIZER_PREFIX=./data/tokenizers/fineweb_48k_sample DATA_PATH=./data/tokens/fineweb_48k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_48k_sample/val D_MODEL=384 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1024 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 TRAIN_MICROBATCH_TOKENS=8192 VAL_BATCH_TOKENS=8192 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=3200 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_s3200.json uv run python train_gpt.py | tee runs/bytelevel48k/bytelevel48k_d384_gqa_softcap_accum_s3200.log
```

Terminal output:

```text
train_batch_size_tokens=16384 train_microbatch_size_tokens=8192 grad_accum_steps=2
parameters=20,499,560
step=0 train_loss=10.8733 train_bpb=3.4118 val_loss=10.8438 val_bpb=3.4832 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=400 train_loss=5.8386 train_bpb=1.8479 val_loss=5.9844 val_bpb=1.9461 muon_lr=3.805e-02 adamw_lr=5.707e-04 elapsed=550.5s
step=800 train_loss=5.1903 train_bpb=1.6323 val_loss=5.6920 val_bpb=1.8215 muon_lr=3.225e-02 adamw_lr=4.838e-04 elapsed=1099.6s
step=1200 train_loss=5.2098 train_bpb=1.6498 val_loss=5.5890 val_bpb=1.7978 muon_lr=2.400e-02 adamw_lr=3.600e-04 elapsed=1648.3s
```

Interpretation:

- The live frontier is no longer the failing full-batch `48k` branch.
- It is now the accumulated `48k d384` branch, which is the strongest denominator test we have found that actually runs locally.
- At `step=400`, the branch was close enough to the older `24k` step-`400` reference (`1.9226`) that it was clearly worth continuing.
- At `step=800`, it is still only `0.0130` bpb behind the older `24k` step-`800` reference (`1.8085`).
- More importantly, it improved `0.1246` bpb from `step=400 -> 800`, slightly better than the `24k` branch's `0.1141` bpb improvement over the same segment.
- So the resized accumulated `48k` branch is still a live candidate rather than a dead end.
- At `step=1200`, it is `0.0222` bpb behind the older `24k` step-`1200` reference (`1.7756`).
- That is a slightly worse relative position than at `step=800`, so the branch now needs a stronger `1200 -> 1600` segment to remain the best active denominator test.

## Experiment 60. Cut The Resized `48k` Branch At `Step=1600`

Status:

- Passed

Purpose:

- Give the healthiest local `48k` configuration one more full segment to prove it could beat the older `24k` family, then stop if it still lagged materially.

Terminal output:

```text
step=1600 train_loss=4.8752 train_bpb=1.5162 val_loss=5.4769 val_bpb=1.7439 muon_lr=1.527e-02 adamw_lr=2.290e-04 elapsed=2197.8s
^C
```

Interpretation:

- The branch stayed stable, but it did not become good enough.
- At `step=1600`, it was `0.0775` bpb worse than the older `24k` `step=1600` reference (`1.6664`).
- That is well outside the local `+/- 0.05` proxy-noise band.
- So denominator scaling stopped being the best active hypothesis, and the device was reassigned to the model-side block ablation.

## Experiment 61. Launch The `24k` Block-Variant Branch

Status:

- In progress

Purpose:

- Test the baseline-inspired `relu2 + block_scales + resid_mix` branch on the strongest tokenizer family we have so far, now that the `48k` denominator path has failed locally.

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_gqa_softcap_relu2_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 MLP_KIND=relu2 USE_BLOCK_SCALES=1 USE_RESID_MIX=1 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_relu2_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_relu2_s1600.log
```

Terminal output so far:

```text
parameters=14,773,384
step=0 train_loss=10.1931 train_bpb=3.3111 val_loss=10.1343 val_bpb=3.4702 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=400 train_loss=5.5333 train_bpb=1.8787 val_loss=5.7186 val_bpb=1.9406 muon_lr=3.251e-02 adamw_lr=4.876e-04 elapsed=477.7s
step=800 train_loss=5.3204 train_bpb=1.7638 val_loss=5.4004 val_bpb=1.8192 muon_lr=1.542e-02 adamw_lr=2.314e-04 elapsed=954.7s
```

Interpretation:

- The new branch starts cleanly and is lighter than the plain `24k` SwiGLU branch.
- At `step=400`, it is `0.0180` bpb worse than the older plain `24k` reference (`1.9226`).
- That is not an immediate win, but it is still inside the local proxy-noise band.
- So this branch deserves at least one more segment before we decide whether the new block controls are helping or hurting.
- At `step=800`, it is `0.0238` bpb worse than the closest old `24k` reference with the same `1600`-step schedule (`1.7954`).
- So it is still slightly behind, but not badly enough to reject yet.

## Experiment 62. Reject The `ReLU^2` Family As The Main Path

Status:

- Passed

Purpose:

- Check whether simplifying the block-variant ablation down to `MLP_KIND=relu2` alone rescued the underperforming all-at-once branch.

Core-only run command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d512_gqa_softcap_relu2_core_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=512 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1365 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 MLP_KIND=relu2 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_relu2_core_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d512_gqa_softcap_relu2_core_s1600.log
```

Terminal output before stop:

```text
parameters=14,771,336
step=0 train_loss=10.1931 train_bpb=3.3111 val_loss=10.1345 val_bpb=3.4702 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=400 train_loss=5.5352 train_bpb=1.8794 val_loss=5.7179 val_bpb=1.9404 muon_lr=3.251e-02 adamw_lr=4.876e-04 elapsed=455.1s
step=800 train_loss=5.3235 train_bpb=1.7648 val_loss=5.3960 val_bpb=1.8177 muon_lr=1.542e-02 adamw_lr=2.314e-04 elapsed=909.9s
^C
```

Interpretation:

- The simplified `relu2` branch behaved almost exactly like the heavier `relu2 + block_scales + resid_mix` branch.
- At `step=800`, it was still `0.0223` bpb worse than the older plain `24k` reference (`1.7954`).
- That is strong evidence that `relu2` itself is not the missing lever here.
- So the `relu2` family is no longer the best use of MPS time.

## Experiment 63. Promote A Wider Plain `24k` Branch

Status:

- In progress

Purpose:

- Revisit the oldest successful empirical signal in the project: more width helped. Test that again in the corrected long-context exact regime on the strongest plain tokenizer branch.

Reference parameter counts:

```text
24k 512 1365 15470216
24k 576 1536 17810072
24k 640 1706 20238248
24k 704 1877 22758392
```

Command:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s1600.log
```

Terminal output so far:

```text
parameters=20,238,248
step=0 train_loss=10.1780 train_bpb=3.3062 val_loss=10.1269 val_bpb=3.4676 muon_lr=2.000e-03 adamw_lr=3.000e-05 elapsed=0.0s
step=400 train_loss=5.3712 train_bpb=1.8237 val_loss=5.5845 val_bpb=1.8951 muon_lr=3.251e-02 adamw_lr=4.876e-04 elapsed=599.1s
step=800 train_loss=5.0739 train_bpb=1.6821 val_loss=5.2047 val_bpb=1.7533 muon_lr=1.542e-02 adamw_lr=2.314e-04 elapsed=1197.8s
```

Interpretation:

- This is the first early checkpoint in quite a while that is actually better than the older plain `24k` reference.
- At `step=400`, it is `0.0275` bpb better than the old plain `24k` `step=400` checkpoint (`1.9226`).
- That makes the wider plain `24k` branch the strongest active hypothesis now.
- At `step=800`, it is `0.0421` bpb better than the closest old `24k` `1600`-schedule reference (`1.7954`).
- That is now close to the edge of the local proxy-noise band in the good direction, so this branch has earned the right to continue toward `step=1600`.

## Experiment 53. Add Baseline-Inspired Block Knobs

Status:

- Passed

Purpose:

- Prepare the next model-side branch without waiting for the live `32k` tokenizer run to finish.
- Add a small set of baseline-inspired block controls so we can test whether more expressive residual mixing and a `ReLU^2` MLP help the exact large-tokenizer regime.

Code added to `train_gpt.py`:

```python
mlp_kind=os.environ.get("MLP_KIND", "swiglu").strip().lower()
use_block_scales=env_bool("USE_BLOCK_SCALES", False)
use_resid_mix=env_bool("USE_RESID_MIX", False)
```

```python
if self.mlp_kind == "swiglu":
    return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))
hidden = torch.relu(self.ff_up(x))
return self.ff_down(hidden * hidden)
```

```python
if self.resid_mix is not None:
    mix = self.resid_mix.to(dtype=x.dtype)
    x = mix[0].view(1, 1, -1) * x + mix[1].view(1, 1, -1) * x0
```

Smoke-test command:

```bash
RUN_ID=baseline_block_smoke DEVICE=cpu MAX_STEPS=1 VAL_LOSS_EVERY=1 VAL_STEPS=1 D_MODEL=64 N_HEADS=8 NUM_KV_HEADS=4 D_FF=128 N_LOOPS=2 SEQ_LEN=32 TRAIN_BATCH_TOKENS=256 VAL_BATCH_TOKENS=256 VOCAB_SIZE=512 TOKEN_DTYPE=uint16 AVG_BYTES_PER_TOKEN=3.5 TIED_EMBEDDINGS=1 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 MLP_KIND=relu2 USE_BLOCK_SCALES=1 USE_RESID_MIX=1 uv run python train_gpt.py
```

Terminal output:

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

- The new knobs compile and run correctly.
- The active `32k` run does not include these changes, because it started before this patch.
- If the live `32k` branch misses `1.5`, these knobs are ready to combine with either the `24k` or `48k` tokenizer path without another refactor cycle.
- This checkpoint also promoted the external reviewer handoff to `DRAFT3.md`.

## Current Working Hypothesis

The best immediate path is now:

1. use the fast PyTorch/MPS trainer as the main local loop
2. keep exact SentencePiece byte accounting enabled so we stay aligned with the official tokenizer semantics
3. treat sampled exact validation as a proxy only, and do not trust differences below about `0.05` bpb
4. use `SEQ_LEN=1024` as the new primary local search regime when the token budget is matched fairly
5. disable QAT during local search and reserve it for final artifact-budget measurement
6. use gradient accumulation when the promising branch is memory-bound rather than accepting an artificially tiny effective batch
7. keep batch large enough for throughput, but compare regimes by total training tokens rather than by raw step count
8. treat the `24k` exact tokenizer path as the current best local scoring branch, but still optimization-limited and artifact-limited rather than architecture-limited
9. longer horizon plus later cooldown is currently the strongest confirmed lever inside the `24k` branch
10. GQA plus query gain plus logit softcap is now the strongest confirmed architecture tweak on top of that schedule
11. longer horizon still helps, but with diminishing returns
12. the `32k` tokenizer branch was informative but no longer worth finishing once it reached `1.6675` at `step=2400`
13. the accumulated `48k d384` branch was the best local denominator test we found, but it still lost clearly by `step=1600`
14. the `relu2` family also underperformed, both in the all-at-once and simplified-core forms
15. the live frontier is now a wider plain `24k` GQA/softcap branch, which is the first recent branch to beat the old `24k` reference at an early checkpoint

Reasoning:

- The official MLX path is correct but too slow on this Mac to be the main search engine.
- The new exact SentencePiece path in `train_gpt.py` is much faster while still respecting official tokenizer byte semantics.
- We now have a concrete exact sampled score trajectory, not just infrastructure.
- The outside review was directionally right that context length needed to be tested much more seriously.
- The follow-up experiments showed that naive step-count comparisons were misleading because the `1024` path needed a smaller batch locally.
- The `24k` exact tokenizer path is now the strongest local branch we have, but it is still bottlenecked by optimization and artifact size more than by missing infrastructure.
- Inside that branch, better LR plus much longer horizon moved the score more than bigger effective batch, extra loops, or extra width.
- On top of that schedule, the official-baseline attention knobs added another clean improvement.
- The `3200`-step continuation shows optimization horizon is still useful, but no longer enough to expect a free drop to `1.5`.
- The next wins are most likely to come from:
  - the active wider plain `24k` branch if extra capacity keeps beating the old reference beyond the first checkpoint
  - then nearby width variants such as `d576` or `d704` if we need to bracket the sweet spot
  - only then reconsider more denominator scaling such as `64k`

## Next Command To Run

The current active command is:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d640_gqa_softcap_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=640 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1706 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d640_gqa_softcap_s1600.log
```

Why this exact command:

- The `48k` denominator path has now had a fair local test and came up short.
- The `relu2` family has also had a fair local test and came up short.
- The model-side recipe is already proven on the `24k` tokenizer family: GQA, query gain, logit softcap, `SEQ_LEN=1024`, long horizon.
- So the clean next question is whether simply giving that exact winning recipe more capacity works better than the failed tokenizer and block detours.

If the current wider `24k` run misses badly, the next command should be:

```bash
PYTHONUNBUFFERED=1 RUN_ID=bytelevel24k_d576_gqa_softcap_s1600 TOKENIZER_PREFIX=./data/tokenizers/fineweb_24k_sample DATA_PATH=./data/tokens/fineweb_24k_sample/train VAL_DATA_PATH=./data/tokens/fineweb_24k_sample/val D_MODEL=576 N_HEADS=8 NUM_KV_HEADS=4 D_FF=1536 N_LOOPS=4 SEQ_LEN=1024 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_TOKENS=16384 VAL_STEPS=16 VAL_LOSS_EVERY=400 MAX_STEPS=1600 COOLDOWN_FRACTION=0.20 QAT_START_FRACTION=1.0 TIED_EMBEDDINGS=1 MUON_LR=0.04 ADAMW_LR=0.0006 QK_GAIN_INIT=1.5 LOGIT_SOFTCAP=30 DEVICE=mps STATS_PATH=runs/bytelevel24k/bytelevel24k_d576_gqa_softcap_s1600.json uv run python train_gpt.py | tee runs/bytelevel24k/bytelevel24k_d576_gqa_softcap_s1600.log
```
