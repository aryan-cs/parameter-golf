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
