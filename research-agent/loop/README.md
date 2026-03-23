# Codex Loop Harness

This folder contains a persistent supervisor loop for Codex. The idea is simple:

- `python3 research-agent/loopctl.py start --config research-agent/loop/config.json` runs forever in the foreground.
- `bash research-agent/start_loop.sh` runs it under `nohup` in the background.
- The controller periodically calls `codex exec`.
- Codex works primarily inside `research-experiments/` and stages manifests in `research-experiments/manifests/pending/`.
- The controller launches those jobs, captures logs, parses simple metrics, and then calls Codex again.
- The controller also opportunistically makes checkpoint commits and pushes when tracked repo files changed, so remote progress is preserved even if a Codex turn forgets to run git itself.

The controller is intentionally external to Codex so progress continues even when an individual Codex turn ends.

## Files

- `research-agent/loop/config.json`: controller settings
- `research-agent/loop/goal.md`: persistent objective and record-track constraints
- `research-agent/loop/prompt_template.md`: the system prompt body used for each Codex turn
- `research-agent/loop/agent_output.schema.json`: structured final message schema for `codex exec`
- `research-agent/loop/run_manifest.example.json`: example job manifest Codex can copy
- `research-agent/loop/runtime/`: controller-managed state, queue, logs, job outputs, and SQLite DB
- `research-experiments/JOURNAL.md`: append-only human-readable research journal updated over time by the loop
- `research-experiments/manifests/pending/`: staged manifests for real jobs

## Commands

Foreground:

```bash
python3 research-agent/loopctl.py start --config research-agent/loop/config.json
```

Single cycle:

```bash
python3 research-agent/loopctl.py once --config research-agent/loop/config.json
```

Background:

```bash
bash research-agent/start_loop.sh
```

Mac `launchd` install:

```bash
bash research-agent/install_loop_launchd.sh
```

Status:

```bash
bash research-agent/status_loop.sh
```

Stop:

```bash
bash research-agent/stop_loop.sh
```

Mac `launchd` remove:

```bash
bash research-agent/uninstall_loop_launchd.sh
```

## Job Manifests

Codex stages manifests in:

- `research-experiments/manifests/pending/`

The controller then ingests ready manifests into:

- `research-agent/loop/runtime/queue/pending/`

Minimal required fields:

- `command`: shell command string

Useful optional fields:

- `job_kind`
- `id`
- `description`
- `cwd`
- `stats_path`
- `env`
- `timeout_seconds`
- `required_python_modules`
- `required_cuda_devices`
- `required_paths`

## Notes

- This harness is record-track oriented by default.
- It uses the locally installed `codex` CLI, not a hidden internal slash command.
- It is designed so the controller keeps running even if a given Codex call finishes quickly.
- The experiment workspace is intentionally separate so `research-experiments/` can hold the actual record-track code, notes, and generated assets without mixing them with supervisor internals.
- The controller distinguishes real `experiment` jobs from support jobs so bootstrap work does not masquerade as a leaderboard result.
