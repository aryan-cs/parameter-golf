# Parameter Golf Workspace

This repository is Aryan Gupta's working Parameter Golf repo: experiment staging, H200 development, and `8xH100` submission prep.

It is not a mirror of the public challenge repository. The goal here is to keep one fast-moving, reproducible workspace centered on the current best lane and the shortest path to an accepted record.

## Current Status

- Primary objective: finish an accepted `10min_16mb` record attempt, not just a good local score.
- Development hardware: `1xH200 NVL`.
- Final validation hardware: `8xH100 SXM`.
- Current recovered lane: `records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/`.
- Best local recovered result so far:
  - `final_int6_sliding_window_exact = 1.11623907`
  - `legal_ttt_exact = 1.11382777`
  - `bytes = 15,860,692`
- Practical acceptance gate against the current public leader:
  - exact `8xH100` seed-1337 run must clear bytes and time caps
  - `legal_ttt_exact <= 1.1144`
  - then the same recipe needs `3` seeds with `p < 0.01`

## Recommended Workflow

If you are picking work back up, use this order:

```bash
python scripts/record_push_status.py --seed 1337
bash scripts/icrn_h200_record_push.sh artifact
bash scripts/icrn_h200_record_push.sh proxy
bash scripts/icrn_h200_record_push.sh combined
bash scripts/icrn_h200_record_push.sh report
```

When `8xH100` access is available, use the exact command printed by `record_push_status.py`. That is the promoted winner from the H200 search state, plus one fallback if needed.

## Repo Map

- `README.md`: this file, focused on the current operating flow.
- `JOURNAL.md`: chronological experiment log and decision history.
- `PLAN.md`: older execution plan; useful for context, not the live source of truth.
- `RUNPOD_READY.md`: current `8xH100` launch checklist and handoff notes.
- `records/`: submission-shaped folders and recovery copies.
- `records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/`: current recovered record lane.
- `scripts/`: launchers, evaluators, packaging helpers, and orchestration.
- `scripts/README.md`: categorized script index.
- `runpod/`: pod sync, launch, watch, and fetch helpers.
- `configs/`: environment and launch config files.
- `candidates/`: older lane snapshots and side paths.
- `data/`: cached dataset and tokenizer assets.

## Current Record Lane

The repo is currently organized around the recovered March 23 public stack plus local H200 continuation work:

- Base stack: LeakyReLU squared MLP, parameter banking, parallel Muon, GPTQ-lite int6, legal score-first TTT.
- Main record-prep folder:
  [`records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback`](records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback)
- Current orchestration entrypoint:
  [`scripts/icrn_h200_record_push.sh`](scripts/icrn_h200_record_push.sh)
- Current search-state reporter:
  [`scripts/record_push_status.py`](scripts/record_push_status.py)

The older VRL/export lane is still in the repo for reference, but it is parked unless the TTT lane fails exact `8xH100` reproduction.

## High-Value Commands

Check the ranked H200 search state:

```bash
python scripts/record_push_status.py --seed 1337
```

Run the full H200-first search sequence:

```bash
bash scripts/icrn_h200_record_push.sh all
```

Run only the cheap artifact-only TTT sweep:

```bash
bash scripts/icrn_h200_record_push.sh artifact
```

Run only the H100-step proxy sweep:

```bash
bash scripts/icrn_h200_record_push.sh proxy
```

Launch the promoted exact `8xH100` seed-1337 candidate:

```bash
ARCH_CANDIDATE=baseline TTT_CANDIDATE=baseline SEED=1337 \
bash scripts/h100_record_push_candidate.sh
```

Package merged metadata from a train log plus resumed legal-TTT log:

```bash
python scripts/prepare_submission_metadata.py LOG_A LOG_B
```

## Conventions

- Keep the repo runnable without reconstructing context from Discord or PR threads.
- Treat `JOURNAL.md` as the experiment ledger.
- Prefer adding submission-shaped outputs under `records/` rather than scattering results.
- Use `scripts/record_push_status.py` as the source of truth for promotion decisions.
- Avoid reopening parked lanes unless the active lane fails on bytes, eval legality, or exact `8xH100` reproduction.

## Related Docs

- [`RUNPOD_READY.md`](RUNPOD_READY.md)
- [`JOURNAL.md`](JOURNAL.md)
- [`scripts/README.md`](scripts/README.md)
- [`records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/README.md`](records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/README.md)
