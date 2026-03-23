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

- Timestamp: 2026-03-23 16:47 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the first sync failure against the stock Runpod image.
- Command or config: Investigated the failed `local_sync_to_pod.sh` attempt and found that the remote image did not include `rsync`.
- Result: Root cause identified: the first sync path assumed `rsync` existed on the pod.
- Decision: Add a `tar`-over-SSH fallback for both sync and fetch so the workflow works on fresh pods without manual package installation.
- Next step: Re-run the sync command against `root@216.243.220.229:16214`, then continue with bootstrap and the first smoke run.

- Timestamp: 2026-03-23 16:53 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the tar-over-SSH ownership warnings seen on the first fallback sync attempt.
- Command or config: Updated the tar sync and fetch paths to ignore ownership and permission preservation, and excluded `.env` from transfer.
- Result: The sync path is now aligned with the pod's container permissions and should stop failing on Mac uid/gid metadata.
- Decision: Retry sync from this machine, then proceed directly to remote bootstrap from the pod.
- Next step: Re-run `bash runpod/local_sync_to_pod.sh root@216.243.220.229 /workspace/golf 16214`, then start `TRAIN_SHARDS=10 bash runpod/pod_bootstrap.sh` over SSH.

- Timestamp: 2026-03-23 17:03 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the first remote launch failure after bootstrap completed.
- Command or config: Inspected `/workspace/golf/runs/.../commit.txt` and traced the failure to `pod_run.sh` assuming the synced pod copy still had a `.git` directory.
- Result: Root cause identified: the sync intentionally excluded `.git`, but the run launcher still required `git rev-parse HEAD`.
- Decision: Stamp the remote sync with a `.sync_commit` file and teach `pod_run.sh` to use it when `.git` is absent.
- Next step: Push the launcher fix, re-sync the repo to `/workspace/golf`, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:08 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the second remote launch failure after the commit stamping fix landed.
- Command or config: Inspected the failed launcher output and found that `uv` was installed under `/root/.local/bin`, but `pod_run.sh` was executing without that directory on `PATH`.
- Result: Root cause identified: non-login SSH shells on the pod do not automatically inherit the `uv` install path.
- Decision: Export `$HOME/.local/bin` in both `pod_bootstrap.sh` and `pod_run.sh` so the workflow is self-contained.
- Next step: Push the path fix, re-sync the repo to `/workspace/golf`, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:16 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the first training-process failure after the launcher reached `torchrun`.
- Command or config: Inspected the failing command path and found that `uv run torchrun` executed the system `torchrun`, which in turn used `/usr/bin/python` outside the synced virtual environment.
- Result: Root cause identified: the launcher was not guaranteeing that distributed training started from the `uv` environment that contained `sentencepiece` and the other synced dependencies.
- Decision: Change the launcher to `uv run python -m torch.distributed.run` and pin `UV_LINK_MODE=copy` to match the pod filesystem behavior.
- Next step: Push the launcher fix, re-sync the repo, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:22 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the next launcher failure after switching away from the system `torchrun`.
- Command or config: Inspected the new failure and found that `uv run python ...` rebuilt a default environment that did not include the `cuda` extra, so `torch` was absent despite bootstrap succeeding earlier.
- Result: Root cause identified: the run launcher should not rely on `uv run` after bootstrap, because the correct CUDA environment already exists at `/workspace/golf/.venv`.
- Decision: Launch training and metadata collection with the bootstrapped `.venv/bin/python` directly.
- Next step: Push the launcher fix, re-sync the repo, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:28 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the launch failure that still reported `torch` missing even after switching to the bootstrapped `.venv`.
- Command or config: Traced the behavior across successive syncs and found that the tar fallback path was deleting `/workspace/golf` before extraction, which erased the pod-side `.venv` and downloaded dataset on every re-sync.
- Result: Root cause identified: the sync helper itself was wiping the environment we were trying to launch from.
- Decision: Stop deleting the remote repo root in the tar fallback path so future syncs preserve `.venv`, data, and runs.
- Next step: Push the sync fix, re-bootstrap the pod once to restore `.venv` and the 10-shard dataset, then relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:36 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Recover from the restored bootstrap failing during CUDA package install on the pod.
- Command or config: Inspected the failure and found `uv sync` hitting a stale file handle while writing into `/workspace/golf/.venv`, which sits on the pod's network-backed workspace volume.
- Result: Root cause identified: the pod virtualenv should not live on the network filesystem.
- Decision: Move the pod virtualenv to local container disk at `/root/.venvs/golf` via `UV_PROJECT_ENVIRONMENT`, and update both bootstrap and launcher scripts to use that path.
- Next step: Push the venv-location fix, re-sync the repo, re-bootstrap the pod, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:47 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Confirm the sync path and clear the next runtime blocker on the first H100 smoke run.
- Command or config: Re-ran `bash runpod/local_sync_to_pod.sh root@216.243.220.229 /workspace/golf 16214`, inspected the failed `train.log`, and traced the crash to `from flash_attn_interface import flash_attn_func as flash_attn_3_func` in `candidates/non_ttt_m22_base/train_gpt.py`.
- Result: The tar-over-SSH sync now completes cleanly from this machine, and the first real training blocker is a missing `flash_attn_interface` module on the stock Runpod PyTorch image.
- Decision: Patch the March 22 base candidate to keep using FlashAttention 3 when available but fall back to PyTorch `scaled_dot_product_attention(..., enable_gqa=...)` when it is not.
- Next step: Commit and push the fallback patch, sync the repo to the pod, verify the candidate imports under the pod venv, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 18:02 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Separate launcher issues from model startup issues after the attention fallback patch landed.
- Command or config: Synced commit `fab56be` to the pod, verified the candidate imports cleanly under `/workspace/golf/.venv`, then ran the March 22 base once through `pod_run.sh` and once directly via `/workspace/golf/.venv/bin/python candidates/non_ttt_m22_base/train_gpt.py` with the same env config.
- Result: The model now starts successfully on the pod and emits the first dataset/tokenizer log lines under direct Python, so the remaining friction is launcher overhead rather than an immediate model crash.
- Decision: Make `runpod/pod_run.sh` launch single-GPU smoke runs with direct Python and reserve `torch.distributed.run` for `NPROC_PER_NODE>1`.
- Next step: Commit and push the launcher change, re-sync the repo, and relaunch the single-H100 smoke run through `pod_run.sh`.

- Timestamp: 2026-03-23 18:06 America/Chicago
- Commit: uncommitted
- Lane: runpod bring-up
- Objective: Clear the next launch failure after switching the smoke path to direct Python.
- Command or config: Re-ran `bash runpod/pod_run.sh non_ttt_m22_base 1337` on the pod and inspected the resulting traceback from `runs/non_ttt_m22_base/.../train.log`.
- Result: The launcher now reaches the script, but it was executing from the per-run output directory, so the candidate's relative `./data/tokenizers/...` path resolved incorrectly and the run failed with `OSError: Not found: "./data/tokenizers/fineweb_1024_bpe.model"`.
- Decision: Keep output artifacts under `runs/...`, but execute training from the repo root so the candidate's relative dataset and tokenizer paths remain valid.
- Next step: Commit and push the working-directory fix, re-sync the repo, and relaunch the March 22 base smoke run.

- Timestamp: 2026-03-23 18:26 America/Chicago
- Commit: uncommitted
- Lane: runpod smoke
- Objective: Capture a useful end-to-end H100 smoke result without wasting repeated bring-up time on long sliding-window eval.
- Command or config: Let the fixed March 22 base run on `1x H100 SXM` through training (`step:810`, `train_time:600548ms`, pre-quant `val_bpb:1.4279`), then interrupted the long post-training sliding-window eval and added `configs/runpod/non_ttt_m22_base_smoke.env` with `EVAL_STRIDE=0`.
- Result: The full training path is now proven on Runpod, the current 10-shard smoke recipe trains at about `0.74s/step`, and the main remaining signal from this smoke run is that post-quant export quality is poor (`final_int6_roundtrip_exact val_bpb:2.72145323`) on this setup.
- Decision: Use the new smoke config for future single-GPU bring-up runs so they complete quickly, and reserve sliding-window final eval for higher-confidence validation runs.
- Next step: Commit and push the smoke config, sync it to the pod, and run `bash runpod/pod_run.sh non_ttt_m22_base 1337 configs/runpod/non_ttt_m22_base_smoke.env`.
