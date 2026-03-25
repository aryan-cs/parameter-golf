# Aryan's Parameter Golf Workspace

This repository is Aryan Gupta's personal workspace for Parameter Golf experiments, submission prep, and infrastructure. It is intentionally a cleaned workspace snapshot, not a mirror of the public challenge repository.

## What Is In This Repo

- `JOURNAL.md`: experiment log and decision history
- `PLAN.md`: working execution plan
- `RUNPOD_READY.md`: launch checklist for cloud reruns
- `configs/runpod/`: environment configs for recovery and launch flows
- `runpod/`: sync, bootstrap, launch, and fetch helpers
- `scripts/`: experiment and packaging helpers
- `candidates/`: local candidate lanes
- `records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/`: current recovered submission-prep folder
- `modal_parameter_golf_h100.py`: Modal launcher for H100 runs

## Current Focus

- keep a reproducible personal workspace for training and packaging
- maintain the recovered under-cap H200 lane
- re-run and validate promising H100 candidates cleanly

## Notes

- public leaderboard history and unrelated imported submission folders have been removed from this repository
- local-only large artifacts such as `final_model.pt` are intentionally ignored
- this repo keeps third-party notices where needed, but the Git history is now intended to reflect Aryan's own workspace only
