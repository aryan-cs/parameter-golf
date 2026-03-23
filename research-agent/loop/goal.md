# Parameter Golf Loop Goal

This repository is in `record-only` mode.

The objective is to keep pushing toward an official `openai/parameter-golf` record submission, not a non-record run.

Operational rules for the loop:

- Work only on ideas that plausibly fit the official record track.
- Prefer modifying or borrowing from actual top record scripts, not the generic root baseline.
- Use the mirrored external winner repo at `research-experiments/cache/external/thwu1-parameter-golf/` as the primary starting point for new experiments when its code is more current or more faithful than our local derived copies.
- Use `research-experiments/cache/external/aruniyer-parameter-golf/` as a secondary reference for third-place ideas, ablations, and implementation details.
- If those mirrors are missing on the current machine, recreate them with `bash scripts/sync_external_public_repos.sh` before relying on them.
- Keep new experiment code, analysis, helper scripts, and outputs inside `research-experiments/` unless you are intentionally changing the automation itself.
- Bootstrap, snapshot, and analysis jobs do not count as leaderboard progress. Only real `job_kind=experiment` runs count.
- If a training or evaluation run is needed, stage a run manifest in `manifests/pending/` with an explicit `job_kind`.
- Do not ask the human to continue the loop unless there is a real blocker.
- Do not treat "good enough for top 3" as success if record-only rules still require beating the current SOTA.
- Keep going by default. The external controller will call Codex again after each turn.
- Keep `JOURNAL.md` up to date as an append-only research log.
- Frequently make small atomic git commits and frequently push them to GitHub so work is preserved remotely, not just locally.

Reality and validity rules:

- Do not use fake, filler, synthetic, toy, proxy, sample, or otherwise non-official data as if it were real leaderboard progress.
- Do not use fabricated metrics, placeholder artifacts, guessed scores, or made-up readiness claims.
- Only treat results as real progress when they come from actual training or evaluation on the official challenge data and tokenizer, with truthful logs and artifacts.
- Infrastructure smoke tests, shortened debug runs, single-GPU pilots, or other submission-invalid runs may be used only to validate plumbing. They must be labeled clearly as infrastructure-only and must never be represented as leaderboard progress.
- The real target is a submission that could actually be uploaded and ranked: official byte budget, official train/eval budgets, reproducible artifacts, real challenge data, and honest current-SOTA comparison.

As of the March 22, 2026 snapshot used to seed this repo:

- leaderboard #1: `1.1428`
- leaderboard #2: `1.1458`
- leaderboard #3: `1.1502`

Important nuance:

- the official repo requires new record submissions to beat the current SOTA by at least `0.005 nats` with sufficient significance
- because of that, leaderboard rank alone is not enough in record-only mode
