# 8xH100 Ready

This file is the handoff checklist for when `8xH100 SXM` time is available.

## Current Objective

Promote one H200-vetted candidate into an exact `8xH100` run, then continue to `3` seeds only if the first seed is submission-viable.

Current practical gate:

- `legal_ttt_exact <= 1.1144`
- artifact under `16,000,000` bytes
- train under `600s`
- eval under `600s`

## Before Booking 8xH100

Run:

```bash
python scripts/record_push_status.py --seed 1337
```

That report is the source of truth for:

- the promoted winner
- the fallback runner-up
- the exact seed-1337 handoff command
- the exact three-seed follow-up command

## First Exact Run

Use the printed winner command from `record_push_status.py`, or the generic form below:

```bash
ARCH_CANDIDATE=baseline TTT_CANDIDATE=baseline SEED=1337 \
bash scripts/h100_record_push_candidate.sh
```

Do not continue to `3` seeds unless that first exact run clears bytes, time, and the `1.1144` score gate.

## If Seed 1337 Clears

Immediately run the exact three-seed wrapper:

```bash
ARCH_CANDIDATE=baseline TTT_CANDIDATE=baseline SEEDS="1337 42 2025" \
bash scripts/h100_record_push_candidate_3seed.sh
```

## If Seed 1337 Misses

Run exactly one seed-1337 repro of the fallback candidate printed by `record_push_status.py`, then reassess instead of opening a wide search on expensive hardware.

## Packaging Checklist

Before calling a run submission-ready, make sure the folder includes:

- `README.md`
- `submission.json`
- train logs
- merged train + resumed-eval metadata if legal TTT was resumed separately
- artifact bytes
- the exact launch command and env
- `3`-seed evidence with `p < 0.01`

Useful helpers:

```bash
python scripts/prepare_submission_metadata.py LOG_A LOG_B
python scripts/summarize_record_runs.py --merge-logs LOG_A LOG_B
```

## Current Working Folder

- Record-prep lane:
  [`records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback`](records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback)
- Orchestrator:
  [`scripts/icrn_h200_record_push.sh`](scripts/icrn_h200_record_push.sh)
- Status report:
  [`scripts/record_push_status.py`](scripts/record_push_status.py)
