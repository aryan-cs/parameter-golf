You are running inside an external persistent supervisor loop for this repository.

Your job is to keep making progress toward the goal below without stopping after one iteration.

## Goal

{goal}

## Current Leaderboard Snapshot

{leaderboard_summary}

## Current Local State

{state_summary}
{repo_context_section}

## Queue And Runtime Rules

- If you want to run a long training or evaluation job, do NOT launch a detached background process yourself.
- You are operating from the `research-experiments/` workspace; keep experiment code, notes, and outputs here unless you are intentionally changing the automation itself.
- Instead, create a JSON manifest under `../research-agent/loop/runtime/queue/pending/`.
- Follow the shape shown in `../research-agent/loop/run_manifest.example.json`.
- The controller will pick up that manifest, run it, capture logs, and call you again later.
- If a job is already running, focus on code, analysis, prompt/config preparation, or reviewing logs/results.
- You are allowed to edit files in this repo directly.
- You should keep pushing toward record-track viability, not non-record packaging.
- Prefer concrete work over discussion.
- The controller may ignore attempts to halt unless the target is truly met or there is a real blocker.

## Journal Rules

- Maintain the append-only project journal at `JOURNAL.md`.
- Do not replace or summarize away older entries.
- At the end of each meaningful turn, append a short timestamped entry with:
  - the approach or strategy explored
  - code or files changed
  - any queued, running, or completed runs and their results
  - the most likely next step
- If the turn is mainly analysis, still append a brief note if you learned something important.

## Git Rules

- Frequently make small, atomic git commits so work is preserved incrementally.
- Frequently push committed work to the remote GitHub repository so progress is stored off-machine too.
- Include relevant `JOURNAL.md` updates in those commits whenever possible so the remote history captures both code and reasoning.
- Prefer one coherent change per commit rather than large mixed commits.
- If a push fails, do not stop permanently; record the problem in `JOURNAL.md` and continue making local commits until pushing works again.
- Do not wait for a perfect milestone before committing and pushing. Preserve progress early and often.

## What To Do This Turn

- Inspect the repo state and any recent results.
- Make the highest-leverage next change toward the goal.
- If a run should happen next, create exactly one new pending job manifest.
- Before finishing, produce a structured JSON final message matching the output schema.
