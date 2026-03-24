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

- Timestamp: 2026-03-23 18:30 America/Chicago
- Commit: uncommitted
- Lane: runpod smoke
- Objective: Keep the next smoke run alive without tying it to an interactive SSH session.
- Command or config: Synced commit `2217c20` to the pod and launched `bash runpod/pod_run.sh non_ttt_m22_base 1337 configs/runpod/non_ttt_m22_base_smoke.env` under `nohup`, with the detached launcher log at `/workspace/golf/logs/non_ttt_m22_smoke_detached_20260323T221010Z.log`.
- Result: A fresh background run is active on the pod at `runs/non_ttt_m22_base/seed1337/20260323T221010Z`, and it has already emitted the standard startup lines through `seed:1337`.
- Decision: Let this detached smoke run continue in the background while the local repo remains free for the next code changes and analysis.
- Next step: Check back on `runs/non_ttt_m22_base/seed1337/20260323T221010Z/train.log` for the completed standard-eval result, then decide whether to move to full-shard repro or a VRL/GPTQ lane.

- Timestamp: 2026-03-23 18:37 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL/GPTQ smoke
- Objective: Open the strongest public non-TTT upside lane under the same single-H100 smoke constraints.
- Command or config: Reviewed `candidates/non_ttt_vrl_gptq/README.upstream.md`, confirmed that the candidate already supports PyTorch SDPA fallback, and added `configs/runpod/non_ttt_vrl_gptq_smoke.env` with `VAL_LOSS_EVERY=0` and `EVAL_STRIDE=0` to keep the 1xH100 proxy run focused on train/final-standard-eval signal.
- Result: The repo now has a fast smoke configuration for the `1.1175` VRL + Full GPTQ lane, ready for an apples-to-apples comparison against the March 22 base on the same Runpod pod and dataset subset.
- Decision: Launch the VRL/GPTQ smoke run next on the existing pod as soon as the base smoke result is captured.
- Next step: Commit and push the VRL smoke config, sync it to the pod, and start `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_smoke.env`.

- Timestamp: 2026-03-23 18:40 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL/GPTQ smoke
- Objective: Start the higher-upside non-TTT lane on the same pod immediately after the March 22 smoke lane finished.
- Command or config: Synced commit `0d7b131` to the pod and launched `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_smoke.env` under `nohup`, with the detached launcher log at `/workspace/golf/logs/non_ttt_vrl_gptq_smoke_detached_20260324T025035Z.log`.
- Result: A fresh background VRL/GPTQ smoke run is active at `runs/non_ttt_vrl_gptq/seed1337/20260324T025035Z`, creating a direct same-pod comparison against the March 22 base smoke results.
- Decision: Let the VRL/GPTQ smoke lane run to completion before deciding whether the next spend should be a full-shard single-GPU repro or an 8xH100 validation.
- Next step: Read `runs/non_ttt_vrl_gptq/seed1337/20260324T025035Z/train.log` after completion and compare both pre-quant and post-quant behavior against the March 22 smoke run.

- Timestamp: 2026-03-23 18:45 America/Chicago
- Commit: uncommitted
- Lane: smoke config hygiene
- Objective: Eliminate two smoke-only evaluation problems that would waste pod time without improving ranking quality.
- Command or config: Added `EXTRA_STRIDE64_FINAL_EVAL` to `candidates/non_ttt_m22_base/train_gpt.py`, set it to `0` in `configs/runpod/non_ttt_m22_base_smoke.env`, and corrected `configs/runpod/non_ttt_vrl_gptq_smoke.env` from `EVAL_STRIDE=0` to `EVAL_STRIDE=2048` so the VRL smoke lane uses a finite non-overlapping final eval.
- Result: Future March 22 smoke runs no longer pay for the forced extra stride-64 tail eval, and the VRL smoke lane will no longer hang at final evaluation because of a zero stride.
- Decision: Push the smoke-eval fixes immediately, then kill the already-launched VRL smoke process and relaunch it with the corrected config.
- Next step: Commit and push these changes, sync them to the pod, restart the VRL/GPTQ smoke run, and compare it against the March 22 smoke baseline.

- Timestamp: 2026-03-23 18:49 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL/GPTQ smoke
- Objective: Replace the first VRL smoke launch with a corrected one after fixing the invalid zero-stride final eval setting.
- Command or config: Synced commit `24253ae` to the pod, stopped the earlier VRL smoke process, and relaunched `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_smoke.env` under `nohup`.
- Result: The corrected VRL smoke lane is now live at `runs/non_ttt_vrl_gptq/seed1337/20260324T025458Z`, using the updated smoke config with `EVAL_STRIDE=2048`.
- Decision: Treat `20260324T025458Z` as the authoritative VRL smoke run and ignore the earlier `20260324T025035Z` attempt for comparisons.
- Next step: Let the corrected run advance through warmup and first logged steps, then compare its pre-quant and post-quant behavior to the March 22 smoke run.

- Timestamp: 2026-03-23 19:07 America/Chicago
- Commit: uncommitted
- Lane: non-ttt comparison
- Objective: Decide which non-TTT lane deserves promotion after both 1xH100 smoke runs completed.
- Command or config: Compared the completed March 22 smoke run (`runs/non_ttt_m22_base/seed1337/20260323T221010Z`) against the completed corrected VRL/GPTQ smoke run (`runs/non_ttt_vrl_gptq/seed1337/20260324T025458Z`) on the same pod and 10-shard proxy setup, then added 8-GPU configs for both lanes.
- Result: March 22 remained slightly better pre-quant (`1.4284` vs `1.4400`), but VRL/GPTQ was decisively better post-quant (`2.3924` vs `2.7577`) while still fitting easily under the size cap (`7.24 MB` vs `6.94 MB` total smoke artifact size).
- Decision: Promote VRL + Full GPTQ to the primary non-TTT candidate for the eventual 8xH100 run, and keep March 22 as the backup lane.
- Next step: Commit and push the 8-GPU run configs, then request an `8x H100 SXM` pod so the promoted VRL lane can be validated on leaderboard-relevant hardware.

- Timestamp: 2026-03-23 19:15 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run prep
- Objective: Get more leaderboard-relevant signal out of the existing single-H100 pod while waiting for 8-GPU hardware.
- Command or config: Started expanding the dataset from 10 to 80 `sp1024` train shards on the pod and added `configs/runpod/non_ttt_vrl_gptq_1gpu_long.env` with `MAX_WALLCLOCK_SECONDS=5400`, `VAL_LOSS_EVERY=1000`, and `EVAL_STRIDE=64`.
- Result: The pod is now downloading the full training shard set, and the repo has a long-run VRL config that should approximate the step count of a 600-second 8xH100 job using a single H100 over about 90 minutes.
- Decision: Once the 80-shard download finishes, launch this long VRL run on the current pod instead of wasting more cycles on tiny-slice proxy runs.
- Next step: Commit and push the long-run config, then queue the long VRL launch behind the active 80-shard download.

- Timestamp: 2026-03-23 19:22 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run prep
- Objective: Automate the handoff from dataset expansion to the longer single-H100 VRL proxy run.
- Command or config: Synced commit `6f8ba66` to the pod and installed a detached watcher at `/tmp/non_ttt_vrl_gptq_1gpu_long_wait_and_run.sh` that polls the shard count every 30 seconds, waits for the `80`-shard download to finish, and then launches `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_1gpu_long.env`.
- Result: The queue watcher is live as PID `128951`, and its log `/workspace/golf/logs/non_ttt_vrl_gptq_1gpu_long_queue_20260324T032220Z.log` is already reporting `waiting shards=65 ...`.
- Decision: Leave the watcher running so the pod automatically rolls into the long VRL run as soon as the dataset download completes.
- Next step: Check back for `80` train shards and confirm that the queued long run has started.

- Timestamp: 2026-03-23 19:30 America/Chicago
- Commit: uncommitted
- Lane: runpod data plumbing
- Objective: Fix the resumed shard downloader after it stalled at 65/80.
- Command or config: Inspected the download log, found `RuntimeError: ... No space left on device`, moved Hugging Face cache policy into `runpod/pod_bootstrap.sh` via `HF_HOME=/workspace/.cache/huggingface`, cleared `/root/.cache/huggingface`, and relaunched the 80-shard downloader with `HF_HOME` and `HUGGINGFACE_HUB_CACHE` pointed at `/workspace`.
- Result: The download resumed successfully from `65` to `80` shards without filling the local root disk.
- Decision: Keep Hugging Face cache on `/workspace` permanently for Runpod workflows so future dataset/bootstrap operations do not silently exhaust the 20 GB root overlay.
- Next step: Let the queue watcher hand off automatically from the completed download to the long VRL run, then monitor the new run.

- Timestamp: 2026-03-23 19:31 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Confirm that the queued long single-H100 VRL run actually started on the full 80-shard dataset.
- Command or config: Read `/workspace/golf/logs/non_ttt_vrl_gptq_1gpu_long_queue_20260324T032220Z.log` after the downloader completed and confirmed that it launched `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_1gpu_long.env`.
- Result: The long run is live at `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z`, and its startup log confirms `train_shards:80`.
- Decision: Let this long VRL run continue; it is now the highest-signal experiment we can run on the current 1xH100 pod.
- Next step: Monitor `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` for the first milestone and compare its eventual quantized score against the earlier 10-shard VRL smoke run.

- Timestamp: 2026-03-23 19:47 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Check whether the fuller-data long proxy run is materially improving the training curve versus the earlier 10-shard smoke runs.
- Command or config: Read `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` after the run reached its first configured validation interval at `step:1000`.
- Result: The 80-shard long run reached `step:1000/20000 val_loss:2.2137 val_bpb:1.3111 train_time:778512ms`, which is a substantial improvement over the earlier 10-shard VRL smoke pre-quant checkpoint (`1.4400` at the 600-second stop).
- Decision: Keep this exact long-run VRL recipe running; it is the strongest evidence so far that the promoted lane is moving toward leaderboard-relevant quality when given a more realistic token budget.
- Next step: Let the long run continue toward its 5400-second stop and inspect the final post-quant score once the GPTQ/export/eval phase completes.

- Timestamp: 2026-03-23 22:53 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Confirm that the promoted long VRL proxy run stayed healthy after the first validation checkpoint.
- Command or config: Sampled the live pod process, GPU telemetry, and `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` while the run continued on the full 80-shard dataset.
- Result: The run is still compute-active on the H100 (`100%` GPU utilization, `24345 MiB` used, about `689 W`) and has advanced to `step:1500/20000 train_loss:2.1386 train_time:1169595ms step_avg:779.73ms` after the earlier `step:1000 val_bpb:1.3111` checkpoint.
- Decision: Leave the run untouched and continue monitoring; the training curve is still improving on the fuller-data proxy and there is no sign of an infrastructure fault.
- Next step: Wait for the next validation milestone and the eventual post-quant export so the current best live pre-quant signal can be compared to a completed quantized score.

- Timestamp: 2026-03-23 23:00 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Measure whether the long VRL proxy keeps improving at the second configured validation checkpoint.
- Command or config: Polled `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` until the run reached `step:2000/20000 val_loss`.
- Result: The full-80-shard run improved again to `step:2000/20000 val_loss:2.1152 val_bpb:1.2527 train_time:1559312ms`, beating its earlier `step:1000 val_bpb:1.3111` checkpoint by another `0.0584` nats.
- Decision: Keep this exact run alive; it is now the strongest training-quality signal we have seen on any lane and is still moving in the right direction.
- Next step: Let the run continue toward its full 5400-second budget, then inspect the final GPTQ/export metrics and decide whether the next pod spend should be another single-GPU ablation or a direct 8xH100 validation.

- Timestamp: 2026-03-23 23:08 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Prepare the most likely next follow-up run without interrupting the active long VRL baseline.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_long_noprune.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_noprune.env`, keeping the full promoted VRL recipe fixed but changing `PRUNE_PCT` from `0.02` to `0.00`.
- Result: The repo now has a ready-to-launch no-prune follow-up for both the long 1xH100 proxy lane and the eventual 8xH100 validation lane.
- Decision: If the current long VRL run still loses too much quality during post-quant export, test the no-prune variant next before changing the architecture or optimizer.
- Next step: Let the active baseline long run finish first, then compare its final quantized score and artifact size against this no-prune contingency plan.

- Timestamp: 2026-03-23 23:12 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Check whether the promoted long VRL proxy continues improving at the third validation checkpoint.
- Command or config: Waited for `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` to reach `step:3000/20000 val_loss`.
- Result: The full-80-shard run improved again to `step:3000/20000 val_loss:2.0758 val_bpb:1.2294 train_time:2339071ms`, with the intermediate training log also confirming `step:2500/20000 train_loss:2.0976`.
- Decision: Keep the current baseline run untouched; it remains the strongest live training signal in the repo and is still improving on schedule.
- Next step: Let the run continue toward its full 5400-second budget, then compare the final quantized output against the prepared no-prune follow-up if export degradation is still the bottleneck.

- Timestamp: 2026-03-23 23:14 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Avoid idle GPU time after the active baseline run finishes on the single-H100 pod.
- Command or config: Copied the new no-prune configs to the pod and queued `/tmp/non_ttt_vrl_gptq_1gpu_long_noprune_after_baseline.sh`, which waits for the baseline PID `129473` to exit and then launches `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_1gpu_long_noprune.env`; queue log: `/workspace/golf/logs/non_ttt_vrl_gptq_1gpu_long_noprune_queue_20260324T041421Z.log`.
- Result: The follow-up watcher is live as PID `135788` and is already polling the active baseline run, so the pod will automatically roll into the no-prune VRL ablation when the current run ends.
- Decision: Keep the baseline run untouched and use the queued no-prune run as the next immediate export-side test if the final quantized score still leaves too much on the table.
- Next step: Let the baseline finish, inspect its final post-quant result, and then compare it directly against the queued no-prune follow-up on the same hardware.

- Timestamp: 2026-03-23 23:26 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Measure whether the active long VRL baseline is still improving at the fourth validation checkpoint.
- Command or config: Waited for `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` to reach `step:4000/20000 val_loss`.
- Result: The run improved again to `step:4000/20000 val_loss:2.0490 val_bpb:1.2136 train_time:3118354ms`, which is now within `0.0136` of the user’s sub-`1.2` target on the single-H100 long proxy.
- Decision: Keep the current baseline run untouched and let it finish; it is now the strongest training-quality signal we have seen so far and is still trending in the right direction.
- Next step: Let the baseline finish its full wallclock budget, inspect the final quantized result, and then use the already-queued no-prune follow-up only if export degradation remains the main gap.

- Timestamp: 2026-03-23 23:43 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Check whether the active long VRL baseline can actually break the sub-`1.2` proxy target before export.
- Command or config: Waited for `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` to reach `step:5000/20000 val_loss`.
- Result: The run improved again to `step:5000/20000 val_loss:2.0080 val_bpb:1.1893 train_time:3897898ms`, which is the first time our live single-H100 long proxy has broken below `1.2`.
- Decision: Keep the current baseline run untouched and let it run all the way through export; we now have direct evidence that this lane is strong enough to justify the queued no-prune export ablation and eventual multi-GPU promotion.
- Next step: Let the baseline finish, capture the final quantized score, and compare it against the queued no-prune follow-up if post-quant degradation is still the main gap.

- Timestamp: 2026-03-23 23:52 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Measure whether the active long VRL baseline keeps improving past the first sub-`1.2` checkpoint.
- Command or config: Waited for `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` to reach `step:6000/20000 val_loss`.
- Result: The run improved again to `step:6000/20000 val_loss:1.9647 val_bpb:1.1636 train_time:4677637ms`, which is our strongest live proxy result so far and materially below `1.2`.
- Decision: Keep the baseline run untouched and let it reach final export/eval; this lane is now clearly strong enough to deserve an eventual 8xH100 promotion once we have the completed post-quant artifact.
- Next step: Let the baseline finish its full wallclock budget, capture the final quantized score, and then compare it against the already-queued no-prune follow-up if export degradation remains the main gap.

- Timestamp: 2026-03-23 23:59 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Track the active baseline through the start of its late training phase while waiting for final export.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` after the `step:6000` validation checkpoint.
- Result: The run has now advanced to `step:6500/20000 train_loss:2.0328`, with `swa:start step:6250` and `late_qat:enabled step:6401 scale:0.1498` appearing in the log, confirming that the final late-training regime is now active.
- Decision: Keep the current baseline untouched and continue waiting for the completed post-quant artifact, because the late-QAT/SWA phase is now in effect and is exactly the part of the recipe we most need to observe through export.
- Next step: Capture the final quantized score from this run, then compare it against the already-queued no-prune follow-up if the export gap is still too large.

- Timestamp: 2026-03-24 00:07 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Capture the strongest late-training checkpoint reached before export begins.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T033051Z/train.log` through the wallclock stop and the start of the export pipeline.
- Result: The run stopped at `step:6926/20000 val_loss:1.9199 val_bpb:1.1370 train_time:5400568ms`, then immediately entered the export path with `ema:applying EMA weights` and `gptq:calibrating with 256 batches...`.
- Decision: Keep the baseline export running to completion; this is now our strongest live signal by a wide margin and close enough to the public frontier that the completed quantized artifact is the most important remaining unknown.
- Next step: Wait for the final post-quant metric from this baseline run, then let the already-queued no-prune follow-up start automatically if the export gap is still too large.

- Timestamp: 2026-03-24 00:13 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Correct the queued follow-up after the baseline export exposed a byte-cap failure mode.
- Command or config: Observed the baseline export log after GPTQ: `prune:zeroed 2288232/26345472 int6 weights (8.7%) threshold=0` and `model:16271727 code:62074 total:16333801 (16.33 MB)`, then added `configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune11.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_prune11.env` with `PRUNE_PCT=0.11`.
- Result: We now have a replacement follow-up that targets the actual problem. The baseline is about `333,801` bytes over the limit, and the previous queued no-prune follow-up would almost certainly have made the artifact larger, not smaller.
- Decision: Replace the queued no-prune follow-up on the pod with the new `prune11` follow-up so the next run spends GPU time on a byte-cap-compliant direction.
- Next step: Sync the new configs, swap the waiting pod script to launch `non_ttt_vrl_gptq_1gpu_long_prune11.env`, and then keep monitoring the current baseline for its final post-quant metric.

- Timestamp: 2026-03-24 00:17 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export monitoring
- Objective: Decide whether the over-cap baseline is still doing useful work or should be terminated in favor of the queued prune-focused follow-up.
- Command or config: Checked the live pod state with `ps`, `nvidia-smi`, remote file mtimes, and the local code path around the export tail in `candidates/non_ttt_vrl_gptq/train_gpt.py`.
- Result: The baseline process is still alive as PID `129473`, using `100%` GPU and about `33 GiB` of memory, while the log has not advanced past `WARNING: Total size 16333801 exceeds 16MB limit by 333801 bytes!`. The code confirms that this warning is printed before the final `final_int6_zstd_roundtrip` evaluation, so the silent tail is still the quantized round-trip eval path rather than a cleanly completed run.
- Decision: Do not kill the baseline yet; let it continue until it either prints the final quantized metric or exits, because that metric is still the most valuable missing datapoint for calibrating the queued prune11 follow-up.
- Next step: Keep monitoring the active baseline for the final quantized metric while leaving the queued `prune11` watcher in place to take over the GPU automatically when this run exits.

- Timestamp: 2026-03-24 00:19 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL pod orchestration
- Objective: Reduce confusion on the pod now that the no-prune path has been superseded by the prune11 follow-up.
- Command or config: Killed the stale helper watchers that were still polling for the old no-prune queue path: remote PIDs `137077` and `140064`.
- Result: The active pod process set is now simpler and matches the intended plan: the long VRL baseline is still running as PID `129473`, and the only queued successor is the prune-focused watcher PID `167816`.
- Decision: Keep the baseline and the prune11 queue intact; remove only obsolete helper processes so the next transition is easier to inspect.
- Next step: Continue monitoring the baseline for a final `final_int6_zstd_roundtrip` metric or process exit, then let the prune11 follow-up take over automatically.

- Timestamp: 2026-03-24 00:21 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL long-run
- Objective: Capture the finished baseline export result and immediately hand the GPU to the prune-focused follow-up.
- Command or config: Monitored the completed baseline log to the end and then attempted a detached launch of `configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune11.env`.
- Result: The finished baseline produced `final_int6_zstd_roundtrip val_loss:1.8842 val_bpb:1.1159 eval_time:624877ms`, which is a strong quantized score but still invalid because the artifact is `16,333,801` bytes. The immediate prune11 relaunch attempt failed because `runpod/pod_run.sh` was still looking for `/root/.venvs/golf/bin/python` even though this pod is using `/workspace/golf/.venv`.
- Decision: Patch the launcher to prefer the repo-local `.venv` so future detached launches and queued follow-ups use the same working Python environment as the successful baseline run.
- Next step: Update `runpod/pod_run.sh`, push the fix, sync it to the pod, and relaunch the prune11 follow-up immediately so the GPU does not sit idle.

- Timestamp: 2026-03-24 00:23 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Restore forward progress after the launcher regression and get the prune-focused follow-up onto the GPU.
- Command or config: Patched `runpod/pod_run.sh` to prefer `$ROOT/.venv` when present, pushed commit `9b01544`, synced the updated launcher to `/workspace/golf/runpod/pod_run.sh`, killed the stale prune11 watcher, and relaunched `bash runpod/pod_run.sh non_ttt_vrl_gptq 1337 configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune11.env`.
- Result: The prune11 run is now live as PID `168452` with run directory `runs/non_ttt_vrl_gptq/seed1337/20260324T052254Z`. Its log has started successfully and confirms the intended recipe change: `VRL Prune(0.11) RawBinary`.
- Decision: Keep prune11 as the active next experiment, since we now know the baseline quality is good enough and the next question is whether slightly stronger pruning can bring the artifact under the cap without giving back too much of the `1.1159` quantized score.
- Next step: Monitor the new prune11 run through its first validation checkpoints and compare its eventual export size and post-quant score against the over-cap baseline.

- Timestamp: 2026-03-24 00:25 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Avoid another idle gap if `prune11` still lands slightly above the byte cap.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_long_prune14.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_prune14.env`, synced them to the pod, and queued `/tmp/non_ttt_vrl_gptq_1gpu_long_prune14_after_prune11.sh` to wait on active prune11 PID `168452` before launching the `1gpu_long_prune14` follow-up.
- Result: The pod now has a second staged fallback ready. Queue log: `/workspace/golf/logs/non_ttt_vrl_gptq_1gpu_long_prune14_queue_20260324T052543Z.log`.
- Decision: Keep `prune11` as the active experiment and treat `prune14` as the next byte-cap hedge rather than jumping straight to a more disruptive architectural change.
- Next step: Monitor prune11 for its first validation checkpoint, then compare its eventual artifact size against the over-cap `1.1159` baseline and let prune14 take over automatically only if needed.

- Timestamp: 2026-03-24 00:27 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export infrastructure
- Objective: Stop paying full 5400-second retrains for export-only pruning changes once the next strong model snapshot exists.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` to support `SAVE_PRE_EXPORT_CHECKPOINT` and `EXPORT_ONLY_CHECKPOINT`, added `maybe_save_pre_export_checkpoint()` plus an `run_export_eval()` path, syntax-checked it with `uv run python -m py_compile`, and updated the queued `prune14` configs to set `SAVE_PRE_EXPORT_CHECKPOINT=1`.
- Result: Future runs can now save a post-EMA full-precision checkpoint before GPTQ/export, and later pruning or export sweeps can start directly from that checkpoint instead of repeating training. The updated script and configs have already been synced to the pod, so the queued `prune14` run will inherit this capability.
- Decision: Keep the active `prune11` run untouched, but use the new export-only infrastructure to accelerate the next round of byte-cap tuning if `prune11` or `prune14` still miss the limit.
- Next step: Push the new export-only infrastructure to GitHub, then monitor `prune11` for its first validation checkpoint while keeping `prune14` as the staged next run.

- Timestamp: 2026-03-24 00:34 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export orchestration
- Objective: Extend the queued compression path so we can sweep a stronger prune setting without paying another full retrain if `prune14` still misses the byte cap.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune17.env`, `configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune17.env`, and `runpod/pod_queue_export_after_prefix.sh`; syntax-checked the helper, synced it plus the new configs to the pod, and launched `bash runpod/pod_queue_export_after_prefix.sh non_ttt_vrl_gptq 1337 non_ttt_vrl_gptq_1gpu_long_prune14 configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune17.env`.
- Result: The pod now has a three-stage queue. `prune11` is active, `prune14` is waiting behind it, and the export-only `prune17` sweep is waiting behind `prune14`. Export queue log: `/workspace/golf/logs/non_ttt_vrl_gptq_1gpu_long_prune14_export_queue_20260324T053401Z.log`.
- Decision: Keep the active training lane unchanged and use the new export-only queue as the next low-latency byte-cap hedge if the saved checkpoint from `prune14` is needed.
- Next step: Monitor `prune11` for its first validation checkpoint and let the staged queue continue automatically unless a clearly better manual intervention appears.

- Timestamp: 2026-03-24 00:35 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Verify that the active `prune11` run is progressing normally and that the staged queue behind it is intact.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T052254Z/train.log`, the active pod process table, and both queue logs for `prune14` and export-only `prune17`.
- Result: `prune11` is healthy on GPU and has progressed through `step:500/20000 train_loss:2.3165 train_time:387941ms`. The pod queue chain is intact: `prune14` is still waiting on prune11 PID `168452`, and the export-only `prune17` helper is waiting for the eventual `prune14` run directory before it launches from the saved checkpoint.
- Decision: Keep the current queue unchanged; there is no reason to interrupt `prune11`, and the staged handoffs now cover both a second full retrain (`prune14`) and a low-latency export-only hedge (`prune17`).
- Next step: Wait for the first `prune11` validation checkpoint at `step:1000`, then compare its quality trend against the baseline to decide whether the byte-cap tradeoff still looks favorable.

- Timestamp: 2026-03-24 00:54 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Reassess whether the active `prune11` run is the best use of GPU time now that its early checkpoints are available.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T052254Z/train.log` through `step:2000`, then manually terminated prune11 PIDs `168452`, `168436`, and `168453` so the queued `prune14` watcher could take over immediately.
- Result: `prune11` reached `step:1000/20000 val_bpb:1.3101` and `step:2000/20000 val_bpb:1.2530`, essentially matching the baseline trajectory. That confirmed the expected behavior: `PRUNE_PCT` only affects export, so the full retrain itself provides no training-side differentiation. After the kill, the queued `prune14` watcher launched the next run as PID `175714` with run directory `runs/non_ttt_vrl_gptq/seed1337/20260324T055414Z`, and its log now shows the intended recipe `VRL Prune(0.14) RawBinary`.
- Decision: Skip the rest of the redundant prune11 retrain and spend the next full training budget on `prune14`, which also saves a reusable pre-export checkpoint for the queued export-only `prune17` sweep.
- Next step: Let `prune14` progress to its first validation checkpoint, then use its saved checkpoint to launch the staged export-only `prune17` sweep if the final artifact is still over the byte cap.

- Timestamp: 2026-03-24 00:55 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export orchestration
- Objective: Confirm that the prune14 cutover is healthy and that the staged export-only sweep is now properly tracking the active source run.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T055414Z/train.log`, checked the active pod processes, and tailed `logs/non_ttt_vrl_gptq_1gpu_long_prune14_export_queue_20260324T053401Z.log`.
- Result: `prune14` is live on GPU as PID `175714` and has progressed through warmup plus the initial training steps with the expected banner `VRL Prune(0.14) RawBinary`. The export-only watcher has moved past “waiting for run dir” and is now logging `source still running for non_ttt_vrl_gptq_1gpu_long_prune14`, which confirms it has detected the active source run and will wait for the saved checkpoint before launching the export-only `prune17` sweep.
- Decision: Keep the current queue intact; the handoff logic is now behaving as intended from active training through the staged export-only hedge.
- Next step: Wait for the first `prune14` validation checkpoint at `step:1000`, then compare it against the earlier trajectory while preparing to consume its saved checkpoint for the queued export-only sweep if the byte cap is still missed.

- Timestamp: 2026-03-24 00:57 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export infrastructure
- Objective: Verify that the active prune14 run is actually capable of producing the reusable checkpoint needed by the staged export-only sweep.
- Command or config: Inspected `runs/non_ttt_vrl_gptq/seed1337/20260324T055414Z/config.env` and `runs/non_ttt_vrl_gptq/seed1337/20260324T055414Z/train_gpt.snapshot.py` on the pod using `grep`.
- Result: The active run bundle confirms `PRUNE_PCT=0.14` and `SAVE_PRE_EXPORT_CHECKPOINT=1`, and the captured trainer snapshot includes `maybe_save_pre_export_checkpoint`, `export_only_checkpoint`, and `run_export_eval`.
- Decision: Keep the export-only `prune17` watcher in place; the currently running prune14 job has the exact checkpoint/export support it needs for that handoff to succeed.
- Next step: Continue monitoring prune14 for its first validation checkpoint and eventual saved checkpoint rather than spending time on more queue surgery.

- Timestamp: 2026-03-24 01:01 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export orchestration
- Objective: Add one more low-latency fallback beyond the staged export-only `prune17` sweep so we can keep moving toward byte-cap compliance without another retrain if needed.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune20.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune20.env`, synced them to the pod, and launched a second export-only watcher with `bash runpod/pod_queue_export_after_prefix.sh non_ttt_vrl_gptq 1337 non_ttt_vrl_gptq_1gpu_export_prune17 configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune20.env`.
- Result: The pod now has a four-stage queue: active `prune14`, then export-only `prune17`, then export-only `prune20`. The new watcher process is live as PID `183233`.
- Decision: Keep the active prune14 run untouched and use the newly staged export-only `prune20` sweep only if `prune17` still misses the byte cap or degrades too much.
- Next step: Wait for the first prune14 validation checkpoint and eventual saved checkpoint, then let the export-only chain consume that checkpoint automatically if the active run still fails the validity bar.

- Timestamp: 2026-03-24 01:04 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export orchestration
- Objective: Extend the export-only fallback chain one more step so the pod can keep pushing toward byte-cap compliance without manual intervention if `prune20` is still slightly too large.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune23.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune23.env`, synced the 1-GPU config to the pod, and launched `bash runpod/pod_queue_export_after_prefix.sh non_ttt_vrl_gptq 1337 non_ttt_vrl_gptq_1gpu_export_prune20 configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune23.env`.
- Result: The pod now has an additional export-only fallback beyond `prune20`. The new watcher is live as PID `184670`, so the staged chain is now: active `prune14`, then export-only `prune17`, then export-only `prune20`, then export-only `prune23`.
- Decision: Keep the active training run unchanged and use `prune23` only if the earlier export-only sweeps still fail the byte cap or lose too much quality.
- Next step: Wait for prune14's first validation checkpoint and saved checkpoint, then let the export-only chain consume that checkpoint automatically until one of the staged pruning levels clears the size cap with acceptable score retention.

- Timestamp: 2026-03-24 01:08 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export validation
- Objective: Verify that the staged export-only sweeps truly bypass retraining and that the current queue depth matches the actual pod processes.
- Command or config: Inspected `candidates/non_ttt_vrl_gptq/train_gpt.py` around the `EXPORT_ONLY_CHECKPOINT` path and checked the live pod process table for `pod_queue_export_after_prefix.sh`.
- Result: The trainer short-circuits exactly as intended when `EXPORT_ONLY_CHECKPOINT` is set: it loads the saved model state, logs `export_only:loaded_checkpoint:...`, calls `run_export_eval(...)`, and returns before optimizer setup or the training loop. The live pod queue now matches the staged design: active `prune14`, then export-only `prune17`, then export-only `prune20`, then export-only `prune23`.
- Decision: Keep the current queue intact; the fastest path to a valid score is to let prune14 produce a reusable checkpoint and consume it through the export-only pruning ladder rather than paying more full retrains.
- Next step: Wait for prune14's first validation checkpoint and eventual saved checkpoint, then let the queued export-only sweeps test progressively stronger pruning until one lands under the 16 MB cap with acceptable score retention.

- Timestamp: 2026-03-24 01:12 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export orchestration
- Objective: Extend the export-only fallback ladder one more step so the pod can keep ratcheting pruning upward automatically if `prune23` still misses the byte cap.
- Command or config: Added `configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune26.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu_export_prune26.env`, synced the 1-GPU config to the pod, and launched `bash runpod/pod_queue_export_after_prefix.sh non_ttt_vrl_gptq 1337 non_ttt_vrl_gptq_1gpu_export_prune23 configs/runpod/non_ttt_vrl_gptq_1gpu_export_prune26.env`.
- Result: The pod now has an additional export-only fallback beyond `prune23`. The new watcher is live as PID `185836`, so the staged chain is now: active `prune14`, then export-only `prune17`, then export-only `prune20`, then export-only `prune23`, then export-only `prune26`.
- Decision: Keep the active training run unchanged and only consume `prune26` if the earlier export-only sweeps are still over the byte cap or lose too much quality.
- Next step: Wait for prune14's first validation checkpoint and eventual saved checkpoint, then let the export-only chain consume that checkpoint automatically until one staged pruning level yields a valid sub-16MB artifact with acceptable score retention.

- Timestamp: 2026-03-24 01:14 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export ablation
- Objective: Compare the first `prune14` validation checkpoint against the earlier strong long-run baseline and confirm the export-only fallback chain is still live behind it.
- Command or config: Monitored `runs/non_ttt_vrl_gptq/seed1337/20260324T055414Z/train.log` through the first validation and checked the live pod process table.
- Result: `prune14` reached `step:1000/20000 val_bpb:1.3093` with `train_loss:2.2667`, which is essentially identical to the earlier long-run baseline at the same point (`1.3111`). The pod queue is still intact behind it: export-only `prune17`, `prune20`, `prune23`, and `prune26` are all waiting in sequence.
- Decision: Keep `prune14` running. This checkpoint confirms the expected behavior that changing `PRUNE_PCT` does not perturb the training trajectory, so the run is still on pace to provide a high-quality checkpoint for the export-only pruning ladder.
- Next step: Let `prune14` continue toward the saved checkpoint and final export, then consume that checkpoint through the staged export-only sweeps until one clears the byte cap with acceptable score retention.

- Timestamp: 2026-03-24 01:20 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL runpod recovery
- Objective: Recover cleanly from a pod connectivity loss without losing the staged export-only orchestration work.
- Command or config: Probed the direct pod endpoint with `ssh` and `nc`, probed the Runpod relay, then added `runpod/pod_queue_export_ladder.sh` and validated it with `bash -n`.
- Result: The direct TCP endpoint began returning `Connection refused`, and the Runpod relay reported `container not found`, which indicates the current pod is no longer available. The new helper script can recreate an entire export-only watcher ladder on a fresh pod with one command instead of restaging each watcher manually.
- Decision: Treat the current live run as interrupted until a replacement pod is available; keep the repo in a ready-to-recover state rather than assuming the old pod will come back.
- Next step: Use the new ladder helper on the next pod to reestablish the staged export-only queue quickly once fresh Runpod access is available.

- Timestamp: 2026-03-24 01:23 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL runpod recovery
- Objective: Reduce fresh-pod recovery to a single launch command so we can restart the active prune14-plus-export ladder quickly after the Runpod outage.
- Command or config: Added `runpod/pod_launch_export_chain.sh`, which starts the source run detached and immediately stages the export-only ladder behind it, then validated it with `bash -n` and a usage-path invocation.
- Result: We now have a one-command pod-side recovery entrypoint that can relaunch `prune14` and queue `prune17 -> prune20 -> prune23 -> prune26` in one shot after `pod_bootstrap.sh` completes.
- Decision: Use the new launcher on the next pod instead of rebuilding the live run and watcher chain by hand.
- Next step: Once a replacement pod exists, sync the repo, run `bash runpod/pod_bootstrap.sh`, then use the new chain launcher to resume the byte-cap sweep path immediately.

- Timestamp: 2026-03-24 01:28 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL runpod recovery
- Objective: Collapse fresh-pod recovery to a single local command so we can resume the active sweep path immediately after the next Runpod launch.
- Command or config: Added `runpod/local_recover_export_chain.sh`, which syncs the repo, runs `pod_bootstrap.sh`, and launches `prune14` plus the full export-only ladder in one SSH session; validated it with `bash -n` and a usage-path invocation.
- Result: Recovery now needs only one local command once a new pod exists. The wrapper targets the exact current plan: `prune14` training plus export-only `prune17 -> prune20 -> prune23 -> prune26`.
- Decision: Use the new local wrapper as the default restart path for the next pod instead of manually issuing sync, bootstrap, and chain-launch commands separately.
- Next step: When a replacement pod is available, run the new wrapper against its SSH target and resume the validity-focused export ladder immediately.

- Timestamp: 2026-03-24 01:34 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Find a byte-cap win that can be implemented locally while Runpod is down, without changing model quality.
- Command or config: Inspected `candidates/non_ttt_vrl_gptq/train_gpt.py` export serialization and patched it so int6 `.q` tensors are serialized as packed 6-bit payloads instead of raw int8 bytes; also updated the loader to unpack the new dtype code during round-trip evaluation. Verified syntax with `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`.
- Result: The active candidate now has a materially denser int6 export format. This should reduce artifact size without changing training or quantization behavior. A full runtime round-trip test is still pending because the local environment currently lacks `torch`, but the patch compiles cleanly and targets the exact part of the pipeline that is causing the validity failure.
- Decision: Keep this serializer improvement in the main lane; it is a direct size win and pairs naturally with the staged prune ladder once we have a fresh pod.
- Next step: Push the serializer change, then rerun the VRL recovery chain on the next pod so we can measure whether packed int6 plus staged pruning clears the 16 MB cap.

- Timestamp: 2026-03-24 01:38 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Add at least one local correctness check for the new packed-int6 serializer even though the current machine lacks `torch` and `numpy`.
- Command or config: Ran a pure-stdlib Python round-trip harness that mirrors the 6-bit grouping logic used by the new serializer and decoder across varied tensor lengths.
- Result: The pure-Python packing check passed for lengths `1, 2, 3, 4, 5, 7, 16, 17, 31, 64, 127, 1024`, which gives us confidence that the bit layout itself is sound. Full runtime verification inside `train_gpt.py` is still pending the next CUDA pod because the local environment does not have `torch`.
- Decision: Keep the packed-int6 serializer patch in place; we now have both a syntax check and an independent bit-layout sanity check.
- Next step: Resume the VRL recovery chain on the next pod and measure the real post-quant artifact size improvement from packed int6 plus the staged prune ladder.

- Timestamp: 2026-03-24 01:43 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Remove additional bookkeeping overhead from the export bundle while preserving compatibility with already-written metadata.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` so quantization metadata now uses compact codes (`'p'`, `'c'`, `6`, `8`) instead of verbose JSON strings and dicts, and switched `json.dumps(...)` to compact separators. Added a small pure-Python compatibility check covering both old and new metadata encodings.
- Result: The export format now wastes fewer bytes on metadata while still accepting older encodings such as `"passthrough"` and `{\"type\":\"int6\"}`. Syntax check passed with `uv run python -m py_compile ...`, and the compatibility check reported `meta kind compatibility ok`.
- Decision: Keep the metadata compaction patch in the main lane; it is low-risk, backward-compatible on load, and directly aligned with the byte-cap problem.
- Next step: Resume the recovery chain on the next pod and measure the combined effect of packed int6 payloads, compact metadata, and staged pruning on final artifact size.

- Timestamp: 2026-03-24 01:50 America/Chicago
- Commit: uncommitted
- Lane: runpod readiness
- Objective: Prepare the repo for immediate restart once new Runpod credits arrive, without needing ad hoc command reconstruction.
- Command or config: Added `runpod/local_recover_export_chain_8gpu.sh` for the `8x H100 SXM` path and added `RUNPOD_READY.md` as a compact runbook covering the current best result, the live lane, and the exact `1x` and `8x` recovery commands. Validated the new wrapper with `bash -n` and a usage-path invocation.
- Result: The repo now has both `1x` and `8x` recovery wrappers plus a single place to look up the current plan the moment credits land. That removes most of the operator friction from going back live.
- Decision: Use the new `8x` wrapper and the runbook as the default restart path once Runpod credits are restored.
- Next step: Wait for credits, then relaunch a pod and resume the validity-focused VRL export ladder immediately.

- Timestamp: 2026-03-24 01:57 America/Chicago
- Commit: uncommitted
- Lane: runpod readiness
- Objective: Add a cheap local guardrail that verifies the restart wrappers and prune ladder still reference a coherent set of configs before we spend fresh credits.
- Command or config: Added `runpod/check_ready.py`, updated `RUNPOD_READY.md` to include the readiness check, and ran `python3 runpod/check_ready.py`.
- Result: The readiness checker passed: both local recovery wrappers reference the expected five configs, the critical prune14 configs still have `SAVE_PRE_EXPORT_CHECKPOINT=1`, and all `1x` / `8x` export-ladder configs are present and point at the VRL trainer.
- Decision: Use `python3 runpod/check_ready.py` as the first local preflight before relaunching any Runpod pod.
- Next step: Wait for credits, run the readiness check, then use the recovery wrapper that matches the pod we bring up.

- Timestamp: 2026-03-24 02:04 America/Chicago
- Commit: uncommitted
- Lane: runpod readiness
- Objective: Make the credit-wait period productive by closing the last operational gap between “pod launched” and “usable monitoring loop.”
- Command or config: Added `runpod/local_watch_latest.sh`, updated `RUNPOD_READY.md` to use it for monitoring, and noted that new Runpod credits are pending approval so the repo should stay in a fully launch-ready state.
- Result: The restart path now has a clean monitor command in addition to sync, bootstrap, launch, and fetch. Once credits land, we can go from zero to a watched run without reconstructing any SSH one-liners by hand.
- Decision: Keep the repo in launch-ready mode while waiting for credits instead of making more speculative code changes.
- Next step: When credits land, run the readiness checker, launch the recovery wrapper for the chosen pod, and monitor it with `runpod/local_watch_latest.sh`.

- Timestamp: 2026-03-24 02:11 America/Chicago
- Commit: uncommitted
- Lane: repo cleanup
- Objective: Clean up the remote repo surface so the root docs and ignore rules reflect the current workflow instead of stale challenge notes and ad hoc ignore patterns.
- Command or config: Rewrote `README.md` as a codebase/operator guide and simplified `.gitignore` into grouped sections for local envs, machine clutter, generated datasets, runs, logs, and Runpod leftovers.
- Result: The root README now explains how to navigate the repo, which lane is active, how the code is organized, and what to run next. The ignore file is shorter, better grouped, and aligned with the current repo layout.
- Decision: Keep the new README as the main entry point for collaborators and use the journal/runbook for evolving operational details.
- Next step: Push the cleanup so the remote repository becomes easier to onboard into before credits land.

- Timestamp: 2026-03-24 02:19 America/Chicago
- Commit: uncommitted
- Lane: runpod readiness
- Objective: Reconfirm that the repo is still launch-ready after the cleanup pass and before new Runpod credits arrive.
- Command or config: Reran `python3 runpod/check_ready.py` and checked the local git state with `git rev-parse HEAD && git status --short`.
- Result: The readiness checker still passes, both recovery wrappers still reference the expected config chains, and the worktree is clean on top of commit `724c979`.
- Decision: Hold the repo steady; there is no new local breakage to fix before a fresh pod is available.
- Next step: Wait for credits, then relaunch immediately from the existing recovery scripts instead of making speculative offline changes.

- Timestamp: 2026-03-24 02:27 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Turn the packed-int6 serializer change into a concrete size estimate so we know whether the next live rerun is likely to clear the byte cap.
- Command or config: Computed the raw storage difference for the logged int6 payload count from the best over-cap run (`26,345,472` int6 values), comparing old int8-byte storage against the new packed-6-bit representation.
- Result: Packed int6 reduces that payload from `26,345,472` raw bytes to `19,759,104` raw bytes, a savings of `6,586,368` bytes before compression. Since the invalid run only missed the final cap by `333,801` bytes, this materially increases confidence that the new export format plus the prune ladder can produce a valid artifact.
- Decision: Treat the packed-int6 serializer as a likely decisive size fix rather than a speculative micro-optimization.
- Next step: Prioritize rerunning the VRL recovery chain with the new serializer as soon as Runpod credits and a fresh pod are available.

- Timestamp: 2026-03-24 02:40 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Remove another source of artifact bloat while credits are still pending by shrinking the export header itself.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` so quantization metadata is stored as a compact binary blob instead of JSON, and so tensor records reference a base-name table with suffix codes instead of storing full tensor names inline. Kept the loader backward-compatible with legacy JSON metadata / legacy tensor headers. Verified syntax with `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and verified the new header codec with a pure-Python round-trip harness for metadata encoding, tensor-ref encoding, and legacy fallback decoding.
- Result: The exporter now removes another layer of duplicated strings from the artifact format while preserving compatibility with older artifacts. This should shave additional bytes off the final packed model on top of the already-implemented packed-int6 payload savings.
- Decision: Keep the compact metadata + tensor-ref table in the main lane; it is a low-risk size optimization that directly targets the remaining validity blocker.
- Next step: Push this change, then relaunch the recovery chain on the next pod so we can measure the combined impact of packed int6, compact metadata, and the staged prune ladder.

- Timestamp: 2026-03-24 02:43 America/Chicago
- Commit: a9eea7b
- Lane: non-ttt VRL export compression
- Objective: Record the remote push for the compact metadata / tensor-ref header change so the journal matches the Git history.
- Command or config: Ran `git commit -m "2026-03-24 runpod: compact export metadata table"` and `git push origin master` after the syntax and pure-Python codec checks passed.
- Result: The exporter header compression work is now on `origin/master` at commit `a9eea7b`, and the worktree is clean again.
- Decision: Keep the repo steady until Runpod credits land; the next meaningful step is a live rerun, not more speculative local churn.
- Next step: Relaunch the recovery chain on a fresh pod and measure whether packed int6 + compact headers + staged prune sweeps finally produce a valid under-16MB artifact.

- Timestamp: 2026-03-24 02:53 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Make the next live rerun more informative by logging an explicit byte breakdown for the artifact instead of only the final compressed size.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` so `run_export_eval` logs `artifact_breakdown` with metadata bytes, tensor-header bytes, packed int6 payload bytes, other payload bytes, raw total bytes, compressed model bytes, and the count of packed int6 tensors. Verified syntax with `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and reran `python3 runpod/check_ready.py`.
- Result: The next live run will tell us exactly where the remaining bytes are going, which will make it much easier to decide whether further pruning is needed or whether another format tweak is the right follow-up. Runpod readiness still reports `OK`.
- Decision: Keep this logging in the main lane; it is low-risk and directly improves the speed of the next debug cycle if the first post-credit rerun is still close to the cap.
- Next step: Push the instrumentation so the remote repo is fully launch-ready before credits land.

- Timestamp: 2026-03-24 03:02 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export readiness
- Objective: Check whether there is any other safe offline improvement left before credits land, and document the actual remaining blocker honestly.
- Command or config: Inspected the tail of `candidates/non_ttt_vrl_gptq/train_gpt.py` to confirm a clean `__main__` guard, then attempted an offline import of the training module via `uv run python` to estimate model/header structure directly. The import failed locally with `ModuleNotFoundError: No module named 'torch'`. Also reran the exporter syntax and readiness checks earlier in the work block.
- Result: There are no more high-confidence offline changes left that can be validated on this machine without a CUDA-capable environment and `torch`. The repo is launch-ready, but real progress from here requires a fresh Runpod pod to measure the live artifact after the packed-int6, compact-header, and artifact-breakdown changes.
- Decision: Stop speculative offline churn and wait for credits / a fresh pod instead of making unvalidated format changes.
- Next step: Relaunch the VRL recovery/export chain on the next pod and use the new artifact breakdown logs to drive any remaining byte-cap sweeps.
