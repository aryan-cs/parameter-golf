# Parameter Golf Workspace

This repository is our working repo for the OpenAI Parameter Golf challenge. The goal is not just to train a strong tiny model, but to produce a **valid** submission:

- total artifact size under `16,000,000` bytes
- training and evaluation compatible with the challenge rules
- competitive `val_bpb` on the FineWeb validation split

At the moment, the main bottleneck is **artifact size**, not model quality.

## Current State

- Best completed quantized score so far: `1.1159 val_bpb`
- Current validity status: **invalid**
- Why invalid: the artifact was `16,333,801` bytes, which is `333,801` bytes over the cap
- Primary active lane: `VRL + Full GPTQ`

If you want the detailed experimental history, read [JOURNAL.md](/Users/aryan/Desktop/golf/JOURNAL.md).

## How To Navigate The Repo

### Root files

- [README.md](/Users/aryan/Desktop/golf/README.md)
  - This file. Start here for repo orientation.
- [JOURNAL.md](/Users/aryan/Desktop/golf/JOURNAL.md)
  - The lab notebook. Every meaningful code change, run, result, and decision should land here.
- [PLAN.md](/Users/aryan/Desktop/golf/PLAN.md)
  - The execution plan for the multi-day search process.
- [RUNPOD_READY.md](/Users/aryan/Desktop/golf/RUNPOD_READY.md)
  - The “credits just landed, launch now” runbook.
- [pyproject.toml](/Users/aryan/Desktop/golf/pyproject.toml)
  - Project dependencies and `uv` setup.

### Training code

- [train_gpt.py](/Users/aryan/Desktop/golf/train_gpt.py)
  - Root baseline / reference entry point.
- [train_gpt_mlx.py](/Users/aryan/Desktop/golf/train_gpt_mlx.py)
  - Apple Silicon / MLX path for local proxy experiments.
- [candidates](/Users/aryan/Desktop/golf/candidates)
  - Active working model lanes.

### Active candidate lanes

- [candidates/non_ttt_vrl_gptq/train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py)
  - Main lane right now.
  - This is the code that produced the strong but currently invalid `1.1159` result.
  - It now includes:
    - packed int6 export payloads
    - compact metadata encoding
    - export-only checkpoint/restart support
- [candidates/non_ttt_m22_base/train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_m22_base/train_gpt.py)
  - March 22 merged-record reference lane.
- [candidates/ttt_lora/train_gpt.py](/Users/aryan/Desktop/golf/candidates/ttt_lora/train_gpt.py)
  - TTT lane reference. Not the current priority.

### Experiment configs

- [configs/runpod](/Users/aryan/Desktop/golf/configs/runpod)
  - Run configs for local promotion, 1x H100 recovery, 8x H100 recovery, and export-only prune sweeps.

Important files there:

- [configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune14.env](/Users/aryan/Desktop/golf/configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune14.env)
  - Main 1x H100 restart config.
- [configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune17.env](/Users/aryan/Desktop/golf/configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune17.env)
- [configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune20.env](/Users/aryan/Desktop/golf/configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune20.env)
- [configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune23.env](/Users/aryan/Desktop/golf/configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune23.env)
- [configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune26.env](/Users/aryan/Desktop/golf/configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune26.env)
  - Export-only prune ladder used to recover validity without paying for more full retrains.

### Runpod automation

- [runpod](/Users/aryan/Desktop/golf/runpod)
  - Everything needed to sync to a pod, bootstrap it, launch runs, queue export sweeps, monitor progress, and fetch artifacts back.

The most important scripts are:

- [runpod/local_sync_to_pod.sh](/Users/aryan/Desktop/golf/runpod/local_sync_to_pod.sh)
  - Sync local repo state to the pod.
- [runpod/pod_bootstrap.sh](/Users/aryan/Desktop/golf/runpod/pod_bootstrap.sh)
  - Install deps with `uv` and prepare the pod.
- [runpod/pod_run.sh](/Users/aryan/Desktop/golf/runpod/pod_run.sh)
  - Run a single config.
- [runpod/pod_launch_export_chain.sh](/Users/aryan/Desktop/golf/runpod/pod_launch_export_chain.sh)
  - Launch the main run and immediately stage export-only sweeps behind it.
- [runpod/local_recover_export_chain.sh](/Users/aryan/Desktop/golf/runpod/local_recover_export_chain.sh)
  - One-command `1x H100` recovery.
- [runpod/local_recover_export_chain_8gpu.sh](/Users/aryan/Desktop/golf/runpod/local_recover_export_chain_8gpu.sh)
  - One-command `8x H100` recovery.
- [runpod/local_watch_latest.sh](/Users/aryan/Desktop/golf/runpod/local_watch_latest.sh)
  - Watch the latest run log from your laptop.
- [runpod/check_ready.py](/Users/aryan/Desktop/golf/runpod/check_ready.py)
  - Local preflight check before spending credits.

### Reference material

- [records](/Users/aryan/Desktop/golf/records)
  - Historical challenge records and non-records.
  - Treat these as references, not as the active lane.
- [data](/Users/aryan/Desktop/golf/data)
  - Dataset tooling and notes.

## What We Are Doing

We are optimizing for one thing: a **valid leaderboard-worthy artifact**.

That breaks into two subproblems:

1. Train a recipe that reaches leaderboard-level quality.
2. Compress/export that recipe so it actually fits under the `16,000,000` byte cap.

We have already solved most of subproblem 1. The active VRL lane reaches strong pre-export and post-quant quality. The current work is focused on subproblem 2.

## How We Are Doing It

### Training strategy

- Use `VRL + Full GPTQ` as the primary lane.
- Use `1x H100` for recovery, bring-up, and cheaper checks.
- Keep `8x H100 SXM` ready for final real runs once credits are available.

### Export strategy

- Prefer changes that do **not** disturb the training curve.
- Save a pre-export checkpoint at the end of a strong run.
- Run export-only prune sweeps from that checkpoint instead of paying for more full retrains.

Current export-side tactics:

- packed int6 payloads
- compact metadata
- export-only prune ladder `17 -> 20 -> 23 -> 26`

### Process strategy

- Use `uv` everywhere.
- Keep [JOURNAL.md](/Users/aryan/Desktop/golf/JOURNAL.md) as the source of truth.
- Commit and push often so the remote repo is always restartable.
- Treat `README` as orientation, `JOURNAL` as history, and `RUNPOD_READY` as the live operator runbook.

## What To Run Next

Before launching a new pod:

```bash
python3 runpod/check_ready.py
```

When credits land and you have a fresh `1x H100` pod:

```bash
bash runpod/local_recover_export_chain.sh root@HOST /workspace/golf PORT 80
```

When you want the real `8x H100 SXM` path:

```bash
bash runpod/local_recover_export_chain_8gpu.sh root@HOST /workspace/golf PORT 80
```

To watch the latest run:

```bash
bash runpod/local_watch_latest.sh root@HOST /workspace/golf PORT non_ttt_vrl_gptq 1337
```

## Source Of Truth

If there is ever a disagreement between files:

- use [JOURNAL.md](/Users/aryan/Desktop/golf/JOURNAL.md) for what actually happened
- use [RUNPOD_READY.md](/Users/aryan/Desktop/golf/RUNPOD_READY.md) for what to do next on Runpod
- use [candidates/non_ttt_vrl_gptq/train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) for the current primary model/export logic

## What “Done” Looks Like

This project is not done when we have a pretty score in a log. It is done when we have:

- a score that is leaderboard-worthy
- an artifact under `16,000,000` bytes
- a run that follows the challenge rules
- and enough reproducible tooling in the repo that we can rerun it cleanly
