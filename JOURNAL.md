# Journal

This file is append-only. Every meaningful code change, run, hypothesis kill, promotion, and rollback goes here immediately.

## Entry Template

- Timestamp:
- Commit:
- Lane:
- Objective:
- Command or config:
- Result:
- Decision:
- Next step:

## Entries

- Timestamp: 2026-03-23 15:55 America/Chicago
- Commit: uncommitted
- Lane: repo
- Objective: Normalize the repo around `uv`, add a durable lab notebook, and wire in the Runpod execution path.
- Command or config: `uv` project setup, Runpod helper scripts, candidate imports, and `PLAN.md` operating rules.
- Result: In progress.
- Decision: Keep the repo runnable from `main` and make the cloud path copy-pasteable before opening more experiment lanes.
- Next step: Finish the `uv` scaffolding, add MLX sliding-window eval, then run verification commands.

- Timestamp: 2026-03-23 16:16 America/Chicago
- Commit: ddd17d3
- Lane: repo / mac-proxy
- Objective: Verify that the new `uv` workflow, Runpod helpers, and MLX sliding-window evaluation are usable.
- Command or config: `uv lock`; `uv sync --extra mlx`; `uv sync --extra cuda`; `uv run python -m py_compile train_gpt_mlx.py runpod/collect_run_metadata.py`; `bash -n runpod/*.sh`; in-memory MLX sanity check with `TRAIN_SEQ_LEN=8` and `EVAL_STRIDE=4`; synthetic `runpod/collect_run_metadata.py` smoke test.
- Result: All checks passed. `uv` resolved both extras cleanly, the shell helpers parsed, MLX standard and sliding eval both returned finite metrics on a toy config, and the Runpod metadata helper wrote the expected JSON summary fields.
- Decision: The repo is ready for the first real cloud smoke launch from `candidates/non_ttt_m22_base`.
- Next step: Sync to Runpod, run `bash runpod/pod_bootstrap.sh`, then launch `NPROC_PER_NODE=1 bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 16:28 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Unblock SSH access for the first Runpod pod launch.
- Command or config: Checked `~/.ssh` for existing public keys and selected the current `id_ed25519.pub` key for Runpod account settings.
- Result: A valid SSH public key already exists locally, so no new keypair is needed.
- Decision: Use the existing `ssh-ed25519` public key in Runpod Settings, then re-enable `SSH terminal access` on the pod deployment form.
- Next step: Paste the public key into Runpod, deploy the `1x H100 SXM` pod, then send back the SSH command from the pod page.

- Timestamp: 2026-03-23 16:39 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Prepare the local sync workflow for the newly provisioned Runpod pod.
- Command or config: Inspected the pod connection panel, selected `SSH over exposed TCP` for direct `rsync` compatibility, and updated the local sync/fetch helpers to accept an optional SSH port.
- Result: The repo can now sync to Runpod using the pod's `root@IP` plus exposed port instead of only a plain `user@host` target.
- Decision: Use the direct TCP SSH target from the pod page for the first bootstrap and run launch.
- Next step: Give the user the exact `ssh`, sync, bootstrap, and launch commands for pod `216.243.220.229:16214`.
