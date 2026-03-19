# Parameter Golf Rules and Source Notes

This file is our local summary of the official OpenAI Parameter Golf competition rules, based on the official repository and record examples.

Status:

- Reviewed against the official repo on March 18, 2026.
- Primary source repo: https://github.com/openai/parameter-golf/tree/main

## What the Competition Is

The main track is:

- best language model under a `16,000,000` byte artifact cap
- trained and evaluated in under `10 minutes` on `8xH100`
- ranked by lowest `val_bpb` on the official FineWeb validation set

The official README describes it as a challenge to train the best LM that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, using tokenizer-agnostic bits-per-byte evaluation.

Important current reference:

- As of March 18, 2026, the public leaderboard top entry is `Naive Baseline`
- exact score in the record metadata: `1.22436570`
- record folder: `records/track_10min_16mb/2026-03-17_NaiveBaseline`

## Main Hard Rules

### 1. Artifact size

The official FAQ says the submission artifact is code bytes plus compressed model bytes.

Interpretation for us:

- the cap is decimal `16,000,000` bytes, not MiB
- the counted code should live in the final submission `train_gpt.py`
- the compressed model is the int8 + zlib roundtrip artifact produced at the end of training
- if we rely on extra code files in a record folder, they are part of what OpenAI will inspect, so the safest path is still a self-contained record snapshot

Official wording we should remember:

> "The submission artifact is computed as code bytes plus compressed model bytes."

### 2. No online access during evaluation

The official FAQ is explicit that evaluation cannot rely on:

- external downloads
- training dataset access
- network calls

That means the final evaluation artifact must be self-contained and reproducible.

Official wording we should remember:

> "No external downloads, training dataset access, or network calls are allowed during evaluation."

### 3. Leaderboard track time limit

For a record submission:

- it must reproducibly run in under `10 minutes` on `8xH100`
- the official baseline uses `MAX_WALLCLOCK_SECONDS=600`
- local Mac runs and 1xH100 runs are for iteration only, not proof of eligibility

### 4. Evaluation flexibility

The official FAQ says:

- evaluation may use any sequence length
- evaluation still must stay under the 10-minute limit on `8xH100`
- evaluation may not access training data unless those bits are somehow included in the artifact budget

### 5. Statistical bar for new records

If we claim a new SOTA record:

- we must beat the current SOTA by at least `0.005 nats`
- we must include enough run logs to show `p < 0.01`
- this significance requirement is waived only for pure systems speedups that do not change the ML

Important nuance:

- the README states the SOTA threshold in `nats`, not `bpb`
- the leaderboard is displayed in `val_bpb`
- for tokenizer changes, OpenAI says submissions will be examined more carefully

Unit conversion we should use in all local score tracking:

- `bpb = nats * log2(e)`
- `0.005 nats = 0.007213475204444817 bpb`
- with the current public leader at `1.22436570`, a record claim should target `1.217152224795555` bpb or better before we treat it as PR-ready

### 6. Tokenizer and dataset changes are allowed, but scrutinized

The official README explicitly allows tokenizer and dataset changes, but with extra burden of proof:

- if we change tokenizer or dataset, we must prove the `val_bpb` calculation is correct
- tokenizer edits will be checked carefully because bugs can falsely improve score
- if we rebuild from official published docs, we should preserve the same document list and ordering

## Official Data and Validation Facts

### 1. Validation split

The official README and `train_gpt.py` agree on this:

- validation is always the full `fineweb_val_*` split
- this is the fixed first `50,000` documents
- local smoke tests that do anything else are only proxies

### 2. Published data workflow

The official repo provides a canonical data workflow under `data/`.

Main pieces:

- `data/cached_challenge_fineweb.py`
- `data/download_hf_docs_and_tokenize.py`
- `data/tokenizer_specs.json`
- `data/README.md`

Canonical local layout from `data/README.md`:

- `data/datasets/<dataset_name>/`
- `data/tokenizers/`
- `data/manifest.json`
- `data/docs_selected.jsonl`
- `data/docs_selected.source_manifest.json`

### 3. Default published baseline dataset

The official starter flow uses:

- `python3 data/cached_challenge_fineweb.py --variant sp1024`
- default published repo id: `willdepueoai/parameter-golf`
- default tokenizer family in `tokenizer_specs.json`: `sp1024`

### 4. Retokenization workflow

If we want a custom tokenizer but still stay aligned with the official document selection:

- rebuild from `docs_selected.jsonl`
- keep the sidecar `docs_selected.source_manifest.json`
- use `download_hf_docs_and_tokenize.py` to export shards from the same selected docs

This matters because OpenAI can verify whether we used the exact same selected document list and order.

## Submission Format

All official submissions are PRs that add a new folder under the appropriate `records/` track.

For a main leaderboard record:

- add a new folder under `records/track_10min_16mb/`

For a non-record under-16MB run:

- add a new folder under `records/track_non_record_16mb/`

For unlimited compute non-record runs:

- still use the non-record track
- note the unlimited-compute nature in the README and metadata

The official README says the PR should only add a new folder under the appropriate `records` subfolder.

## Required Files in a Submission Folder

The official README requires:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- any other dependencies needed for the record snapshot to run

The official record examples include exactly:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`

## `submission.json` Shape

From the official baseline and non-record examples, these fields are expected:

```json
{
  "author": "Your Name",
  "github_id": "your-github-id",
  "name": "Run Name",
  "blurb": "Short description",
  "date": "2026-03-18T00:00:00Z",
  "val_loss": 0.0,
  "val_bpb": 0.0,
  "bytes_total": 0,
  "bytes_code": 0
}
```

Optional fields seen in the non-record example:

- `track`
- `pre_quant_val_loss`
- `pre_quant_val_bpb`
- `step_stop`
- `wallclock_seconds`
- `bytes_model_int8_zlib`

## What the Official Baseline Actually Does

The baseline record tells us what "legit" looks like today:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- tied embeddings enabled
- global batch `524288` tokens per step
- wallclock cap `600` seconds
- periodic full validation

Key baseline numbers from the official record:

- exact metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
- total artifact size: `15,863,489` bytes
- code size: `47,642` bytes
- serialized model int8+zlib: `15,815,847` bytes

## Constraints on Root Starter Files

The official root `train_gpt.py` and `train_gpt_mlx.py` are starter scripts, not the place for final SOTA logic.

The file header in the official `train_gpt.py` says:

- the root scripts are "good launching-off points"
- competitive submissions should stay in `/records`
- both `train_gpt.py` and `train_gpt_mlx.py` must never exceed `1500` lines

Current official line counts in the repo snapshot I reviewed:

- root `train_gpt.py`: `1126` lines
- root `train_gpt_mlx.py`: `1088` lines

## What This Means for Our Repo Right Now

### 1. Our current experiments are proxy experiments

Our local repo currently:

- trains on small locally packed samples
- uses our own trainer, tokenizer, packer, and autopilot helpers
- does not yet use the official distributed CUDA baseline path
- does not yet produce a real `records/...` submission folder

So:

- our local scores are useful for search
- our local scores are not leaderboard-comparable yet

### 2. Before claiming a competition-valid result, we must align these pieces

We will need:

- official validation behavior on the fixed 50k-doc split
- a record-folder snapshot under `records/...`
- a self-contained record `train_gpt.py`
- a real `train.log`
- a real `submission.json`
- an artifact-size check that matches the official code path
- proof that any custom tokenizer/export computes `val_bpb` correctly

### 3. Our strongest local discipline going forward

Before we call any run "competition relevant," we should be able to answer:

- Which official track is it for?
- What exact validation split is it using?
- Does it fit under `16,000,000` bytes after int8 + zlib roundtrip?
- Can it run under `600` seconds on `8xH100`?
- Is the submission packaged as a valid `records/...` folder?
- If the tokenizer changed, can we prove the bpb calculation is correct?

## Recommended Working Checklist

This is the checklist we should use before moving further:

1. Keep local proxy experiments for fast search, but label them as proxy only.
2. Read and mirror the official `records/` submission layout.
3. Decide whether we are targeting `track_10min_16mb` or a non-record track first.
4. Move from our custom local dataset pipeline toward the official published-doc workflow.
5. Treat custom tokenizer work as high-risk until we can prove exact `val_bpb` correctness.
6. Only call something "record-worthy" if it clears the artifact cap, timing cap, and packaging requirements.

## Source Files Reviewed

Primary sources used for this file:

- Official repo README: https://github.com/openai/parameter-golf/blob/main/README.md
- Official data README: https://github.com/openai/parameter-golf/blob/main/data/README.md
- Official starter trainer: https://github.com/openai/parameter-golf/blob/main/train_gpt.py
- Official baseline record README: https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
- Official baseline metadata: https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json
- Official non-record example: https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md
