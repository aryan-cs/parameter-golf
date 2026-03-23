# Parameter Golf Journal

This is the append-only project journal for overnight Codex work on the `openai/parameter-golf` record track.

## 2026-03-22 21:25 CDT - Journal Bootstrap

- Canonical journal filename set to `JOURNAL.md`.
- Loop controller, prompt template, and goal file all point at this file.
- Record-only mode is enabled. The controller treats beating third place as insufficient; the true target is a valid new SOTA run under the official rules.
- Git discipline is part of the operating prompt: frequent small atomic commits and frequent pushes to GitHub.

## 2026-03-22 21:26 CDT - Restored Prior Context

- A bootstrap research job previously completed with `val_bpb=1.1556`.
- The current public leaderboard snapshot tracked by the controller is:
- `#1 1.1428`
- `#2 1.1458`
- `#3 1.1502`
- The last recorded Codex turn timed out before structured output, so the service is being restarted from a clean state.

## 2026-03-22 21:23 CDT - Rename Verified And Loop Restarted

- The configured journal path was normalized to uppercase `JOURNAL.md`.
- The background `launchd` service `com.parameter-golf.codex-loop` was reinstalled and restarted successfully.
- Live status now reports controller PID `18395` in the running state.
- Heartbeat is updating under `loop/runtime/heartbeat.json`, confirming the controller is actively polling.

## 2026-03-22 21:29 CDT - Workspace Split Into Agent And Experiments

- Moved the persistent supervisor, prompts, runtime DB, queue, and launch scripts into `research-agent/`.
- Moved the append-only journal and live experiment bootstrap script into `research-experiments/`.
- Updated the controller config so Codex now runs from `research-experiments/` and writes the journal at `research-experiments/JOURNAL.md`.
- Retargeted the queued official-record refresh job to run from the new experiments workspace.

## 2026-03-23T02:29:05.989672+00:00 - Controller Job Launched: refresh_official_top_records_score_parser_20260323

- Description: Refresh the official top-record snapshot after fixing score extraction and train-script discovery.
- Command: PYTHONDONTWRITEBYTECODE=1 python3 scripts/bootstrap_official_top_records.py
- CWD: /Users/aryan/Desktop/golf/research-experiments
- Manifest: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/running/refresh_official_top_records_score_parser_20260323.json
- Log: /Users/aryan/Desktop/golf/research-agent/loop/runtime/jobs/refresh_official_top_records_score_parser_20260323/run.log
- Timeout seconds: 1800
- PID: 20776

## 2026-03-23T02:29:36.069482+00:00 - Controller Job Result: refresh_official_top_records_score_parser_20260323

- Status: done
- Exit code: 0
- Description: Refresh the official top-record snapshot after fixing score extraction and train-script discovery.
- Command: PYTHONDONTWRITEBYTECODE=1 python3 scripts/bootstrap_official_top_records.py
- val_bpb: 1.1502
- artifact_bytes: n/a
- train_time_ms: n/a
- Manifest: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/done/refresh_official_top_records_score_parser_20260323.json
- Log: /Users/aryan/Desktop/golf/research-agent/loop/runtime/jobs/refresh_official_top_records_score_parser_20260323/run.log

## 2026-03-22 21:34 CDT - Controller Auto Push Enabled

- Added controller-side git checkpoint support so tracked changes can still be committed and pushed even if an individual Codex turn forgets to run git.
- The controller now checks for repo changes after job polling, after job launches, and after Codex turns.
- The configured remote remains `origin`, which points at `https://github.com/aryan-cs/parameter-golf.git`.
- This should make the GitHub repo reflect ongoing research progress more reliably during unattended runs.
