# Parameter Golf Loop Goal

This repository is in `record-only` mode.

The objective is to keep pushing toward an official `openai/parameter-golf` record submission, not a non-record run.

Operational rules for the loop:

- Work only on ideas that plausibly fit the official record track.
- Prefer modifying or borrowing from actual top record scripts, not the generic root baseline.
- Keep new experiment code, analysis, helper scripts, and outputs inside `research-experiments/` unless you are intentionally changing the automation itself.
- If a training or evaluation run is needed, create a run manifest in `../research-agent/loop/runtime/queue/pending/`.
- Do not ask the human to continue the loop unless there is a real blocker.
- Do not treat "good enough for top 3" as success if record-only rules still require beating the current SOTA.
- Keep going by default. The external controller will call Codex again after each turn.
- Keep `JOURNAL.md` up to date as an append-only research log.
- Frequently make small atomic git commits and frequently push them to GitHub so work is preserved remotely, not just locally.

As of the March 22, 2026 snapshot used to seed this repo:

- leaderboard #1: `1.1428`
- leaderboard #2: `1.1458`
- leaderboard #3: `1.1502`

Important nuance:

- the official repo requires new record submissions to beat the current SOTA by at least `0.005 nats` with sufficient significance
- because of that, leaderboard rank alone is not enough in record-only mode
