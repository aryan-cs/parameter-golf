# 7-Day Runpod-First Execution Plan

Goal: end Day 7 with one submission-ready recipe that fits under `16,000,000` bytes, is reproducible on `8xH100 SXM`, and is competitive with the live public frontier.

This is an execution document. The default action is to launch the next informative run, not to brainstorm.

## Core Strategy

- Lane A is the dependable non-TTT path starting from the March 22 merged record stack in `candidates/non_ttt_m22_base`.
- Lane B is the public non-TTT upside path in `candidates/non_ttt_vrl_gptq`.
- Lane C is the TTT upside path in `candidates/ttt_lora`, opened only after the base model is stable.
- The Mac is a proxy lane for ranking, debugging, and export sanity only. Real search moves onto Runpod as soon as the code path is clean.

## Permanent Operating Rules

- Use `uv` everywhere.
- Local Apple Silicon setup: `uv sync --extra mlx`.
- Runpod CUDA setup: `uv sync --extra cuda`.
- Every Python command runs through `uv run`.
- Commit after every material code change.
- Commit before every Runpod launch.
- Commit after every completed run batch that changes our direction.
- Push to `origin` (`https://github.com/aryan-cs/parameter-golf.git`) at least every 2 hours during active work.
- Never end a work block, sleep, or handoff with unpushed progress.
- Keep `main` runnable. Use lane branches only when parallel changes would otherwise conflict.
- `JOURNAL.md` is append-only and updated immediately after every code change that affects experiments, every completed run, every killed hypothesis, and every promotion or rollback decision.
- Every run bundle under `runs/` must contain the env/config, launch command, log file, artifact sizes, train wallclock, eval wallclock, post-quant `val_bpb`, and the related commit SHA.

## Daily Cadence

Every day follows the same loop:

1. Review the last completed run bundle and the last `JOURNAL.md` entry.
2. Decide the next 1-3 runs only.
3. Commit and push the code that launches them.
4. Launch immediately.
5. While compute is busy, make only the smallest high-ROI code change for the next batch.
6. End the day with overnight work already running.

## 7-Day Sequence

### Day 1

Objective: make the repo operable from both the Mac and Runpod.

- Normalize the repo around `uv`.
- Add the Runpod helper scripts and `configs/runpod/*.env`.
- Add `JOURNAL.md` and start logging immediately.
- Upgrade `train_gpt_mlx.py` with sliding-window eval support.
- Launch the first `1xH100` smoke run from `non_ttt_m22_base`.

### Day 2

Objective: establish a trustworthy non-TTT H100 baseline.

- Reproduce `non_ttt_m22_base` cleanly on Runpod.
- Fix only blockers or obvious reproduction bugs.
- Promote the first trustworthy `8xH100` baseline run.

### Day 3

Objective: improve export quality on the strongest non-TTT lane.

- Run narrow ablations only:
  - GPTQ-lite clip search or export alignment
  - EMA vs tight SWA settings
  - warmdown `3000` vs `3500`
  - fp16 or mixed-bit placement if bytes allow

### Day 4

Objective: open the stronger public non-TTT upside lane.

- Bring up `non_ttt_vrl_gptq` on cloud.
- Promote it immediately if post-quant `val_bpb` beats the base and artifact size still fits.

### Day 5

Objective: open the TTT lane without blocking the base path.

- Keep the stronger non-TTT lane running.
- Start `ttt_lora`.
- Kill the TTT lane quickly if eval wallclock or byte compliance looks bad.

### Day 6

Objective: run decisive validations.

- Pick the two strongest contenders.
- Run decisive `8xH100` validations with clean logs and submission-shaped outputs.
- Require stability, artifact compliance, and complete run bundles.

### Day 7

Objective: freeze the winner and package the final submission.

- Pick the winner by score, stability, and reproducibility.
- Rerun once if cleaner logs or a tighter artifact are needed.
- Finalize the record-style folder and make sure all code, configs, notes, and run metadata are committed, pushed, and journaled.

## First Queue

1. `uv` project setup and lockfile.
2. Runpod bootstrap and run scripts.
3. MLX sliding-window eval implementation and sanity check.
4. `1xH100` smoke run on `non_ttt_m22_base`.
5. `1xH100` follow-up on the stronger of the base or first ablation.

## What Not To Do

- Do not spend hours polishing the Mac proxy.
- Do not open TTT before the non-TTT path is healthy.
- Do not mix many changes into one run when one clean change can answer the question.
- Do not leave progress only on disk; every meaningful state change should be committed, pushed, and journaled.
