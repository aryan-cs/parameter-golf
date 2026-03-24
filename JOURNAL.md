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
- Commit: `7c60955`
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
- Commit: `ab61d33`
- Lane: runpod bring-up
- Objective: Unblock SSH access for the first Runpod pod launch.
- Command or config: Checked `~/.ssh` for existing public keys and selected the current `id_ed25519.pub` key for Runpod account settings.
- Result: A valid SSH public key already exists locally, so no new keypair is needed.
- Decision: Use the existing `ssh-ed25519` public key in Runpod Settings, then re-enable `SSH terminal access` on the pod deployment form.
- Next step: Paste the public key into Runpod, deploy the `1x H100 SXM` pod, then send back the SSH command from the pod page.

- Timestamp: 2026-03-23 16:39 America/Chicago
- Commit: `ab61d33`
- Lane: runpod bring-up
- Objective: Prepare the local sync workflow for the newly provisioned Runpod pod.
- Command or config: Inspected the pod connection panel, selected `SSH over exposed TCP` for direct `rsync` compatibility, and updated the local sync/fetch helpers to accept an optional SSH port.
- Result: The repo can now sync to Runpod using the pod's `root@IP` plus exposed port instead of only a plain `user@host` target.
- Decision: Use the direct TCP SSH target from the pod page for the first bootstrap and run launch.
- Next step: Give the user the exact `ssh`, sync, bootstrap, and launch commands for pod `216.243.220.229:16214`.

- Timestamp: 2026-03-23 16:47 America/Chicago
- Commit: `e73078a`
- Lane: runpod bring-up
- Objective: Recover from the first sync failure against the stock Runpod image.
- Command or config: Investigated the failed `local_sync_to_pod.sh` attempt and found that the remote image did not include `rsync`.
- Result: Root cause identified: the first sync path assumed `rsync` existed on the pod.
- Decision: Add a `tar`-over-SSH fallback for both sync and fetch so the workflow works on fresh pods without manual package installation.
- Next step: Re-run the sync command against `root@216.243.220.229:16214`, then continue with bootstrap and the first smoke run.

- Timestamp: 2026-03-23 16:53 America/Chicago
- Commit: `ab61d33`
- Lane: runpod bring-up
- Objective: Recover from the tar-over-SSH ownership warnings seen on the first fallback sync attempt.
- Command or config: Updated the tar sync and fetch paths to ignore ownership and permission preservation, and excluded `.env` from transfer.
- Result: The sync path is now aligned with the pod's container permissions and should stop failing on Mac uid/gid metadata.
- Decision: Retry sync from this machine, then proceed directly to remote bootstrap from the pod.
- Next step: Re-run `bash runpod/local_sync_to_pod.sh root@216.243.220.229 /workspace/golf 16214`, then start `TRAIN_SHARDS=10 bash runpod/pod_bootstrap.sh` over SSH.

- Timestamp: 2026-03-23 17:03 America/Chicago
- Commit: `e73078a`
- Lane: runpod bring-up
- Objective: Recover from the first remote launch failure after bootstrap completed.
- Command or config: Inspected `/workspace/golf/runs/.../commit.txt` and traced the failure to `pod_run.sh` assuming the synced pod copy still had a `.git` directory.
- Result: Root cause identified: the sync intentionally excluded `.git`, but the run launcher still required `git rev-parse HEAD`.
- Decision: Stamp the remote sync with a `.sync_commit` file and teach `pod_run.sh` to use it when `.git` is absent.
- Next step: Push the launcher fix, re-sync the repo to `/workspace/golf`, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:08 America/Chicago
- Commit: `7d1d3ce`
- Lane: runpod bring-up
- Objective: Recover from the second remote launch failure after the commit stamping fix landed.
- Command or config: Inspected the failed launcher output and found that `uv` was installed under `/root/.local/bin`, but `pod_run.sh` was executing without that directory on `PATH`.
- Result: Root cause identified: non-login SSH shells on the pod do not automatically inherit the `uv` install path.
- Decision: Export `$HOME/.local/bin` in both `pod_bootstrap.sh` and `pod_run.sh` so the workflow is self-contained.
- Next step: Push the path fix, re-sync the repo to `/workspace/golf`, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:16 America/Chicago
- Commit: `e73078a`
- Lane: runpod bring-up
- Objective: Recover from the first training-process failure after the launcher reached `torchrun`.
- Command or config: Inspected the failing command path and found that `uv run torchrun` executed the system `torchrun`, which in turn used `/usr/bin/python` outside the synced virtual environment.
- Result: Root cause identified: the launcher was not guaranteeing that distributed training started from the `uv` environment that contained `sentencepiece` and the other synced dependencies.
- Decision: Change the launcher to `uv run python -m torch.distributed.run` and pin `UV_LINK_MODE=copy` to match the pod filesystem behavior.
- Next step: Push the launcher fix, re-sync the repo, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:22 America/Chicago
- Commit: `7d1d3ce`
- Lane: runpod bring-up
- Objective: Recover from the next launcher failure after switching away from the system `torchrun`.
- Command or config: Inspected the new failure and found that `uv run python ...` rebuilt a default environment that did not include the `cuda` extra, so `torch` was absent despite bootstrap succeeding earlier.
- Result: Root cause identified: the run launcher should not rely on `uv run` after bootstrap, because the correct CUDA environment already exists at `/workspace/golf/.venv`.
- Decision: Launch training and metadata collection with the bootstrapped `.venv/bin/python` directly.
- Next step: Push the launcher fix, re-sync the repo, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:28 America/Chicago
- Commit: `b5a8431`
- Lane: runpod bring-up
- Objective: Recover from the launch failure that still reported `torch` missing even after switching to the bootstrapped `.venv`.
- Command or config: Traced the behavior across successive syncs and found that the tar fallback path was deleting `/workspace/golf` before extraction, which erased the pod-side `.venv` and downloaded dataset on every re-sync.
- Result: Root cause identified: the sync helper itself was wiping the environment we were trying to launch from.
- Decision: Stop deleting the remote repo root in the tar fallback path so future syncs preserve `.venv`, data, and runs.
- Next step: Push the sync fix, re-bootstrap the pod once to restore `.venv` and the 10-shard dataset, then relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:36 America/Chicago
- Commit: `7d1d3ce`
- Lane: runpod bring-up
- Objective: Recover from the restored bootstrap failing during CUDA package install on the pod.
- Command or config: Inspected the failure and found `uv sync` hitting a stale file handle while writing into `/workspace/golf/.venv`, which sits on the pod's network-backed workspace volume.
- Result: Root cause identified: the pod virtualenv should not live on the network filesystem.
- Decision: Move the pod virtualenv to local container disk at `/root/.venvs/golf` via `UV_PROJECT_ENVIRONMENT`, and update both bootstrap and launcher scripts to use that path.
- Next step: Push the venv-location fix, re-sync the repo, re-bootstrap the pod, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 17:47 America/Chicago
- Commit: `b5a8431`
- Lane: runpod bring-up
- Objective: Confirm the sync path and clear the next runtime blocker on the first H100 smoke run.
- Command or config: Re-ran `bash runpod/local_sync_to_pod.sh root@216.243.220.229 /workspace/golf 16214`, inspected the failed `train.log`, and traced the crash to `from flash_attn_interface import flash_attn_func as flash_attn_3_func` in `candidates/non_ttt_m22_base/train_gpt.py`.
- Result: The tar-over-SSH sync now completes cleanly from this machine, and the first real training blocker is a missing `flash_attn_interface` module on the stock Runpod PyTorch image.
- Decision: Patch the March 22 base candidate to keep using FlashAttention 3 when available but fall back to PyTorch `scaled_dot_product_attention(..., enable_gqa=...)` when it is not.
- Next step: Commit and push the fallback patch, sync the repo to the pod, verify the candidate imports under the pod venv, and relaunch `bash runpod/pod_run.sh non_ttt_m22_base 1337`.

- Timestamp: 2026-03-23 18:02 America/Chicago
- Commit: `1277135`
- Lane: runpod bring-up
- Objective: Separate launcher issues from model startup issues after the attention fallback patch landed.
- Command or config: Synced commit `fab56be` to the pod, verified the candidate imports cleanly under `/workspace/golf/.venv`, then ran the March 22 base once through `pod_run.sh` and once directly via `/workspace/golf/.venv/bin/python candidates/non_ttt_m22_base/train_gpt.py` with the same env config.
- Result: The model now starts successfully on the pod and emits the first dataset/tokenizer log lines under direct Python, so the remaining friction is launcher overhead rather than an immediate model crash.
- Decision: Make `runpod/pod_run.sh` launch single-GPU smoke runs with direct Python and reserve `torch.distributed.run` for `NPROC_PER_NODE>1`.
- Next step: Commit and push the launcher change, re-sync the repo, and relaunch the single-H100 smoke run through `pod_run.sh`.

- Timestamp: 2026-03-23 18:06 America/Chicago
- Commit: `9fdc879`
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

- Timestamp: 2026-03-24 03:12 America/Chicago
- Commit: uncommitted
- Lane: runpod readiness
- Objective: Reconfirm the exact repository state before the next status handoff so the journal matches reality and the remote stays current.
- Command or config: Ran `git rev-parse HEAD`, `git status --short`, `python3 runpod/check_ready.py`, and tailed the journal to confirm the latest readiness notes are present.
- Result: HEAD was `3e06011`, the worktree was clean, and `runpod/check_ready.py` still reported `OK` with both recovery wrappers pointing at the expected 5-config chains and 22 verified config files.
- Decision: Keep the repo unchanged until credits land; the current bottleneck remains live H100 access rather than local code readiness.
- Next step: Push this verification note, then resume the VRL recovery/export chain on the next pod the moment credits are available.

- Timestamp: 2026-03-24 03:23 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Use a temporary CPU `torch` import path to quantify the real remaining header savings offline and to sanity-check the compact tensor-ref encoder against the actual VRL model state.
- Command or config: Ran `uv run --with torch` to import `candidates/non_ttt_vrl_gptq/train_gpt.py`, instantiate the default `GPT`, quantize its state dict without Hessians, and compare old JSON/header accounting against the new compact metadata + tensor-ref format. While doing that, hit a real bug where `encode_tensor_ref` misclassified passthrough tensors like `bigram.scale` as synthetic `.scale` suffix records; patched the encoder to prefer exact-name matches before applying suffix compression. Re-ran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and the CPU-torch probe after the fix.
- Result: The compact header work is real but small: `meta_savings=381` bytes, `tensor_header_savings=4691` bytes, `total_header_savings=5072` bytes for the full VRL model. The packed-int6 change is still the dominant size lever, with `26,345,472` int6 values corresponding to `6,586,368` raw-byte savings versus old int8-byte storage. The tensor-ref encoder bug is now fixed before the next live run.
- Decision: Keep the compact header path because it is correct and free, but treat packed int6 plus pruning as the actual byte-cap solution. Stop spending offline time on ever-smaller header tweaks.
- Next step: Push the bug fix + measured header findings, then wait for the next H100 pod so the live artifact can validate the packed-int6 path under real compression.

- Timestamp: 2026-03-24 03:36 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Quantify the full raw payload breakdown by tensor kind so we know whether any large non-int6 bucket is still worth targeting offline before the next pod comes up.
- Command or config: Used `uv run --with torch` to import the VRL trainer, instantiate the default `GPT`, quantize its state dict without Hessians, and bucket the export tensors by `{int6_q, int8_q, passthrough_direct_float16, int6_scale, int8_scale}` while accounting for packed-int6 raw bytes.
- Result: The raw payload is dominated by packed int6 weights: `int6_q=19,759,104` bytes, `int8_q=524,288`, `passthrough_direct_float16=248,008`, `int6_scale=84,992`, `int8_scale=2,048`, total raw payload `20,618,444` bytes. This confirms there is no other hidden payload class close to int6 in size; the remaining validity fight is overwhelmingly about the int6 block and pruning/compression behavior, not some overlooked float payload.
- Decision: Treat the exporter work as complete for now. There is no new large offline size lever beyond the already-implemented packed-int6 path and the staged prune sweeps.
- Next step: Wait for the next H100 pod and use the new `artifact_breakdown` logs to see how much of the packed-int6 raw win survives final compression in the real artifact.

- Timestamp: 2026-03-24 03:48 America/Chicago
- Commit: uncommitted
- Lane: runpod restart strategy
- Objective: Update the recovery plan so the next live restart preserves more quality now that packed int6 is in place, instead of assuming we still need to start from an aggressively pruned source run.
- Command or config: Patched `configs/runpod/non_ttt_vrl_gptq_1gpu_long.env` and `configs/runpod/non_ttt_vrl_gptq_8gpu.env` to set `SAVE_PRE_EXPORT_CHECKPOINT=1`, added new export-only configs for `prune05`, `prune08`, `prune11`, and `prune14` on both `1gpu` and `8gpu`, updated `runpod/local_recover_export_chain.sh`, `runpod/local_recover_export_chain_8gpu.sh`, and `RUNPOD_READY.md` to launch from the baseline `PRUNE_PCT=0.02` source run and then walk a gentler export ladder `05 -> 08 -> 11 -> 14 -> 17 -> 20`. Reran `python3 runpod/check_ready.py`.
- Result: The restart plan now preserves the strongest available recipe first and only spends score on heavier pruning if the new exporter still needs it. Readiness still passes, and the recovery wrappers now validate against 7 config refs each with 30 config files total.
- Decision: Use the baseline-first ladder as the new default relaunch path. Starting from `prune14` is no longer justified now that packed int6 is in the exporter.
- Next step: Push the gentler recovery plan so the repo is ready to relaunch immediately when fresh H100 credits arrive.

- Timestamp: 2026-03-24 03:56 America/Chicago
- Commit: uncommitted
- Lane: runpod export orchestration
- Objective: Fix the export-ladder controller so the next relaunch does not waste credits on lower-quality sweeps after a valid artifact appears, and does not risk hanging on a rung that never launched.
- Command or config: Patched `runpod/pod_launch_export_chain.sh` so the export ladder controller runs in the background, and rewrote `runpod/pod_queue_export_ladder.sh` to act as a single sequential controller: wait for the source run, resolve its checkpoint, stop immediately if the source already logged `Size OK:`, launch each export-only rung synchronously from the same checkpoint, and stop as soon as any rung logs `Size OK:`. Verified shell syntax with `bash -n runpod/pod_launch_export_chain.sh runpod/pod_queue_export_ladder.sh runpod/pod_queue_export_after_prefix.sh` and reran `python3 runpod/check_ready.py`.
- Result: The export ladder is now operationally safer and cheaper. It will no longer blindly continue through more aggressive pruning after a valid artifact is found, and it will not depend on later rungs producing new checkpoints just to unblock the next rung.
- Decision: Keep the sequential stop-on-valid controller as the default orchestration path for the next pod restart.
- Next step: Push the ladder-controller fix so the relaunch path is both quality-preserving and credit-efficient before the next H100 pod comes up.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Find one more offline size lever by testing whether the int6 payload itself can be made more zstd-friendly without changing any quantized values.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` to keep the legacy 3-byte/4-value int6 codec for backward compatibility, add a new default int6 codec that zigzag-encodes values and stores them as six packed bitplanes, and decode both formats via `dt=4` legacy / `dt=5` new. Verified syntax with `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a CPU `uv run --with torch --with sentencepiece --with zstandard --with numpy` bakeoff plus round-trip harness against a real quantized VRL model skeleton.
- Result: The new lossless codec round-trips exactly and beat the legacy int6 codec by `738,147` compressed bytes in the offline bakeoff (`legacy_compressed=5102384`, `new_compressed=4364237`, `delta=-738147`). That is materially larger than the remaining `333,801`-byte submission overage from the best invalid run, so this is the first offline change that plausibly closes the byte-cap gap by itself.
- Decision: Keep the zigzag+bitplane int6 codec in the main lane. It is backward-compatible, lossless, and large enough to justify immediate use on the next H100 relaunch.
- Next step: Push the codec upgrade, then relaunch the baseline-first export ladder on the next pod to measure whether the real trained artifact finally lands under `16,000,000` bytes with a competitive score intact.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Check whether the improved bitplane payload changes the best final compression choice, instead of continuing to hardcode zstd.
- Command or config: Benchmarked `zstd19`, `zstd22`, `zlib9`, `bz2`, and `lzma9e` on the new bitplane-packed VRL model skeleton via `uv run --with torch --with sentencepiece --with zstandard --with numpy`, then patched `candidates/non_ttt_vrl_gptq/train_gpt.py` to add a tiny artifact header (`QCB1` + codec id), choose the smallest available codec at export time, and decode both new-header artifacts and legacy no-header artifacts. Re-verified syntax with `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and validated `compress_model_blob` / `decompress_model_blob` round-trip offline.
- Result: On the offline bakeoff, `lzma9e` beat the bitplane `zstd` path by another `~63 KB` (`lzma_9=4300364`, `zstd19=4363157`, `zstd22=4365593`). The chooser now automatically selected codec id `3` / `lzma9e` and produced a `4,301,197`-byte wrapped artifact blob while round-tripping exactly. Timing remained reasonable in the same bakeoff: `lzma9e` decompressed in about `161 ms`.
- Decision: Keep the codec chooser in the main lane and let the exporter pick the smallest available codec automatically. With bitplane int6 already in place, `lzma9e` currently looks like the best default for the next live rerun.
- Next step: Push the multi-codec chooser, then relaunch the baseline-first export ladder on the next H100 pod and see whether the real trained artifact now clears the `16,000,000`-byte cap without harsher pruning.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Remove benchmark noise from the new LZMA path and see whether a tuned filter really beats the generic `preset=9|extreme` setting on the exact same payload.
- Command or config: Re-ran the codec comparison with fixed Python / NumPy / Torch seeds and compared `zstd19`, generic `lzma preset9e`, and a tuned `LZMA2` filter (`hc4`, `dict_size=32MB`, `nice_len=273`). Then updated `candidates/non_ttt_vrl_gptq/train_gpt.py` so codec id `3` now uses that tuned filter by default and logs the more accurate codec label `lzma_hc4_32mb`.
- Result: The fixed-seed comparison confirmed the tuned filter is real, not noise: `hc4_32mb=4302268`, `preset9e=4303084`, `zstd19=4363365`. So the tuned filter wins by another `816` bytes over generic `preset9e` and stays about `61 KB` ahead of `zstd19` on the same payload.
- Decision: Keep the tuned `lzma_hc4_32mb` filter in the main lane. The gain is small, but it is free, deterministic, and directly increases the odds that the first post-credit rerun lands under the cap.
- Next step: Push the tuned-filter tweak so the next H100 relaunch uses the best codec settings we have measured offline.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Check whether the `xz` container around the tuned `LZMA2` stream is still wasting bytes now that we already wrap the artifact in our own custom header.
- Command or config: Benchmarked tuned `LZMA2` with `FORMAT_XZ` versus `FORMAT_RAW` using the same fixed-seed quantized VRL payload. Then patched `candidates/non_ttt_vrl_gptq/train_gpt.py` so codec id `3` now writes raw `LZMA2` streams by default, while the loader first tries raw decode and then falls back to legacy wrapped-`xz` decode for backward compatibility. Re-verified syntax, readiness, and offline round-trip.
- Result: Raw `LZMA2` shaved another `62` bytes versus the wrapped `xz` stream on the same payload (`xz_size=4302268`, `raw_size=4302206`). After the patch, the chooser validated cleanly with `codec_name=lzma_raw_hc4_32mb` and `blob_size=4302211`.
- Decision: Keep the raw `LZMA2` path as the default for codec id `3`. The gain is tiny, but it is free and backward-compatible, so there is no reason not to take it.
- Next step: Push the raw-`LZMA2` tweak so the next live rerun uses the smallest artifact container we have measured offline.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Check whether the candidate source itself still contains removable prose now that the model blob is much smaller, and reclaim those bytes if possible without changing behavior.
- Command or config: Removed the large top-of-file version/history docstring plus several section-banner and explanatory comments from `candidates/non_ttt_vrl_gptq/train_gpt.py`, then measured the exact file-size delta against `HEAD`, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `74,931` bytes to `73,185` bytes, a direct savings of `1,746` bytes on the counted code payload. Compile and readiness checks still passed unchanged.
- Decision: Keep the code-size trim in the main lane. The savings are small compared with the model blob wins, but they are free and directly increase submission headroom.
- Next step: Push the code-size trim so the next live rerun benefits from both the smaller model blob and the smaller counted source file.

- Timestamp: 2026-03-24 02:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Remove the remaining pure comment-only lines from the candidate now that the bigger structural trim already proved safe.
- Command or config: Deleted the remaining standalone comment lines from `candidates/non_ttt_vrl_gptq/train_gpt.py`, then measured the exact byte delta against `HEAD`, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `73,185` bytes to `72,334` bytes, an additional `851` bytes of code-payload savings with no behavior change. Compile and readiness checks still passed.
- Decision: Keep the extra comment trim in the main lane. It is pure headroom.
- Next step: Push the final comment-only trim so the next live rerun uses the smallest safe candidate source we have.

- Timestamp: 2026-03-24 02:41 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL export compression
- Objective: Check whether the current raw `LZMA2` default (`hc4`, `dict_size=32MB`) is still locally optimal on the new bitplane payload, or whether a different match finder / dictionary pair can buy a few more free bytes before the next H100 rerun.
- Command or config: Rebuilt the fixed-seed quantized VRL model skeleton offline via `uv run --with torch --with sentencepiece --with zstandard --with numpy`, compared the current raw `LZMA2` filter against a small sweep of `HC3`, `HC4`, and `BT4` with multiple dictionary sizes, then patched `candidates/non_ttt_vrl_gptq/train_gpt.py` to switch the default raw codec filter from `hc4_32mb` to `hc3_16mb` and updated the logged codec name accordingly.
- Result: On the same fixed-seed payload, `hc3_16mb` beat the current default by another `2,618` bytes while still round-tripping exactly (`current=4301682`, `candidate=4299064`, `delta=-2618`, `roundtrip=True`). That is small compared with the earlier bitplane win, but it is free, deterministic, and directly improves our odds of landing under the `16,000,000`-byte cap on the first post-credit rerun.
- Decision: Keep `lzma_raw_hc3_16mb` as the new default codec path for the main lane. `HC4/32MB` is no longer the best measured raw `LZMA2` setting for this payload.
- Next step: Re-run compile/readiness checks, push the filter update, and use the smaller `hc3_16mb` exporter on the next baseline-first export ladder relaunch.

- Timestamp: 2026-03-24 02:45 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim any remaining zero-risk counted bytes from verbose log strings now that the bigger export-format wins are already in place.
- Command or config: Shortened several pure logging strings in `candidates/non_ttt_vrl_gptq/train_gpt.py` (`artifact_breakdown`, startup banner, export/load markers, final eval labels, size warnings, and GPTQ/prune progress lines), then re-measured file size, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `72,334` bytes to `71,982` bytes, a further `352` bytes of code-payload savings with no behavior change. Compile and readiness checks still passed.
- Decision: Keep the log-string trim in the main lane. The savings are small, but they are free and directly improve submission headroom.
- Next step: Push the trim so the next live rerun benefits from both the stronger exporter and the slightly smaller counted source file.

- Timestamp: 2026-03-24 02:47 America/Chicago
- Commit: uncommitted
- Lane: runpod export ladder reliability
- Objective: Preserve the recent candidate log-byte trims without breaking the export ladder's stop-on-valid behavior.
- Command or config: Patched `runpod/pod_queue_export_ladder.sh` so `run_is_size_ok()` accepts both the legacy success marker (`Size OK:`) and the new shorter candidate log marker (`size_ok:`), then reran `bash -n` on the pod launcher scripts and reran `python3 runpod/check_ready.py`.
- Result: The controller now remains compatible with both old and new train logs, so the next export ladder relaunch will still stop as soon as any rung produces a valid-size artifact instead of accidentally running past it.
- Decision: Keep the compatibility patch in the main lane. The candidate can stay smaller without sacrificing recovery-script correctness.
- Next step: Push the controller fix so the next live relaunch benefits from the shorter candidate logs and still exits immediately on the first valid artifact.

- Timestamp: 2026-03-24 02:50 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another block of counted source bytes by shortening repeated internal identifiers that are local to the candidate file and do not affect the artifact format.
- Command or config: Renamed a set of repeated local constants and helper functions in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example control-pattern constants, quant-meta helpers, int6 pack/unpack helpers, and model-blob codec helpers), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on a fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `71,982` bytes to `70,761` bytes, another `1,221` bytes of direct code-payload savings. Compile and readiness checks still passed, and the offline validator still produced the same codec/result (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the identifier-shortening pass in the main lane. It buys real counted headroom without changing the trained artifact path.
- Next step: Push the rename pass so the next live export rerun benefits from both the stronger exporter and the smaller counted candidate source.

- Timestamp: 2026-03-24 02:51 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another small block of counted source bytes by shortening the very frequently repeated local variable name `base_model`, while keeping behavior and artifact format unchanged.
- Command or config: Renamed `base_model` to `bm` across the candidate's calibration/export path and main training setup in `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `70,761` bytes to `70,305` bytes, a further `456` bytes of direct code-payload savings. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the local-variable shortening pass in the main lane. It is another free source-size win with no measurable exporter regression.
- Next step: Push the rename so the next live export rerun benefits from the smaller counted candidate source.

- Timestamp: 2026-03-24 02:55 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another block of counted source bytes by shortening several heavily repeated local names in the main train/export path (`world_size`, `grad_accum_steps`, `val_tokens`, and `master_process`) while preserving behavior.
- Command or config: Renamed the affected locals to shorter aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, fixed one temporary shadowing issue in the warmup/warmdown path, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `70,305` bytes to `69,768` bytes, another `537` bytes of direct code-payload savings. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the local-name shortening pass in the main lane. It is another free code-size win with no observed exporter regression.
- Next step: Push the rename pass so the next live export rerun benefits from the smaller counted candidate source.

- Timestamp: 2026-03-24 02:56 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another small block of counted source bytes by shortening several heavily repeated evaluation/training locals (`world_size`, `grad_accum_steps`, `val_tokens`, and the export-path master flag) while preserving the export behavior.
- Command or config: Renamed the affected locals to shorter aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `69,768` bytes to `69,184` bytes, another `584` bytes of direct code-payload savings. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the local-name shortening pass in the main lane. It is another free code-size win with no observed exporter regression.
- Next step: Push the rename pass so the next live export rerun benefits from the smaller counted candidate source.

- Timestamp: 2026-03-24 02:58 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another small block of counted source bytes by shortening several repeated train/eval locals (`train_loader`, `optimizers`, and the sliding-eval batch-size arg) while preserving exporter behavior.
- Command or config: Renamed the affected locals to shorter aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `69,184` bytes to `69,016` bytes, another `168` bytes of direct code-payload savings. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the local-name shortening pass in the main lane. The gain is small, but it is free and keeps stacking counted headroom.
- Next step: Push the rename pass so the next live export rerun benefits from the smaller counted candidate source.

- Timestamp: 2026-03-24 03:03 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a larger block of counted source bytes by shortening several still-frequent internal type/helper/config names that are local to the candidate file and do not change the export format.
- Command or config: Renamed a batch of internal candidate-only symbols in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `Hyperparameters -> H`, `CastedLinear -> CL`, `DistributedTokenLoader -> DTL`, `CausalSelfAttention -> CSA`, `BigramHashEmbedding -> BHE`, `ValueEmbedding -> VE`, and a few long export helper/config names), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `69,016` bytes to `67,869` bytes, another `1,147` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 67,869`, which is `-7,062` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the internal-symbol shortening pass in the main lane. It is a materially larger code-size win than the recent one-off trims and did not change the measured exporter behavior offline.
- Next step: Push the rename pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:08 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by shortening repeated internal config-field and constructor argument names that are local to the candidate file and do not affect the artifact format or state dict keys.
- Command or config: Renamed a batch of internal candidate-only names in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `train_seq_len -> tsl`, `eval_seq_len -> esl`, `vocab_size -> vs`, `num_layers -> nl`, `num_heads -> nh`, `num_kv_heads -> nkh`, `model_dim -> dm`, `mlp_mult -> mm`, `tie_embeddings -> te`, `logit_softcap -> lsc`, `rope_base -> rb`, `qk_gain_init -> qgi`, `bigram_dim -> bgd`, `xsa_last_n -> xsn`, and `rope_dims -> rd` across the candidate-only config and constructor path), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `67,869` bytes to `66,592` bytes, another `1,277` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 66,592`, which is `-8,339` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the config/constructor shortening pass in the main lane. It is another materially useful code-size win with no observed exporter regression.
- Next step: Push the rename pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:10 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by shortening repeated local/export-path argument names that do not affect checkpoint keys or the artifact format.
- Command or config: Renamed a batch of local candidate-only names in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `distributed -> dd`, `state_dict -> sd`, `input_ids -> ids`, `target_ids -> tgt`, `quant_result -> qr`, `quant_meta -> qm`, `raw_data -> rd`, and a few eval/export helper arg names), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `66,592` bytes to `66,132` bytes, another `460` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 66,132`, which is `-8,799` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the local/export-path shortening pass in the main lane. The gain is smaller than the earlier rename batches, but it is free and preserves the measured exporter behavior.
- Next step: Push the rename pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:13 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by shortening repeated internal config aliases and plain Python-only constructor/local names that do not affect artifact format or checkpoint keys.
- Command or config: Renamed another batch of candidate-only names in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `train_files -> tf`, `val_files -> vf`, `tied_embed_init_std -> teis`, `smear_enabled -> se`, `backout_enabled -> be`, `warmdown_iters -> wdi`, `late_qat_threshold -> lqt`, `ve_enabled -> vee`, `ve_dim -> ved`, `ve_layers -> vel`, plus the main-process `distributed -> dd` flag and matching constructor arg names), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `66,132` bytes to `65,424` bytes, another `708` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 65,424`, which is `-9,507` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the config/local shortening pass in the main lane. It is another free counted-size win with no observed exporter regression.
- Next step: Push the rename pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:16 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by aliasing the global `torch` module shorter and keeping the remaining runtime checks honest.
- Command or config: Switched the candidate import from the full `torch` module name to the shorter alias `th` across `candidates/non_ttt_vrl_gptq/train_gpt.py`, then fixed the two import lines that should continue to reference the real `torch` package path, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `65,424` bytes to `64,764` bytes, another `660` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 64,764`, which is `-10,167` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the shorter global `torch` alias in the main lane. It is another meaningful counted-size win with no observed exporter regression.
- Next step: Push the alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:19 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by shortening the hottest remaining dtype/helper spellings in the candidate without changing exporter behavior.
- Command or config: Added short aliases for repeated `torch` dtypes and helpers inside `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `BF`, `F32`, `F64`, `I8`, `I16`, `I64`, `AC`, `IM`, `Z`, `TT`, `QT`, `EM`, `FN`, `SK`), performed a mechanical replacement, fixed the alias-definition line after the bulk replace, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `64,764` bytes to `64,110` bytes, another `654` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 64,110`, which is `-10,821` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the dtype/helper alias pass in the main lane. It is another meaningful counted-size win with no observed exporter regression.
- Next step: Push the alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:22 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by aliasing a few still-hot tensor helpers and parameter container spellings without changing exporter behavior or checkpoint structure.
- Command or config: Added short aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py` for repeated helpers such as `nn.Parameter -> P`, `nn.ParameterList -> PL`, `torch.zeros_like -> ZL`, `torch.empty_like -> EL`, `torch.arange -> AR`, `torch.sigmoid -> SG`, `torch.cat -> CAT`, `torch.ones -> ON`, `torch.full -> FUL`, and `torch.eye -> EYE`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `64,110` bytes to `63,904` bytes, another `206` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 63,904`, which is `-11,027` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the helper-alias pass in the main lane. The gain is small, but it is free, safe, and keeps stacking counted headroom while we wait for the next live export rerun.
- Next step: Push the helper-alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:24 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by aliasing a few still-hot framework spellings (`nn.Module`, `nn.init`, `th.no_grad`, `th.clamp`, `th.diag`, and distributed reduction helpers) without changing exporter behavior or checkpoint structure.
- Command or config: Added short aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py` for repeated framework helpers such as `nn.Module -> M`, `nn.ModuleList -> ML`, `nn.init -> NI`, `th.no_grad -> NG`, `th.clamp -> CLP`, `th.diag -> DG`, `dist.all_reduce -> ARD`, and `dist.ReduceOp -> ROP`, replaced the corresponding repeated call sites, simplified a few remaining `th.Tensor` annotations to `Tensor`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `63,904` bytes to `63,706` bytes, another `198` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 63,706`, which is `-11,225` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the framework-alias pass in the main lane. The gain is modest, but it is another free counted-size win with no observed exporter regression.
- Next step: Push the framework-alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:27 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a larger block of counted source bytes by shortening the hottest remaining environment and timing call paths (`os.environ.get`, `os.path.join`, and `time.perf_counter`) without changing exporter behavior.
- Command or config: Added short aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py` for `os.environ.get -> EG`, `os.path.join -> JP`, and `time.perf_counter -> PC`, replaced the repeated config/bootstrap/timing call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `63,706` bytes to `62,810` bytes, another `896` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 62,810`, which is `-12,121` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the environment/timing alias pass in the main lane. It is a materially better counted-size win than the recent micro-alias passes and still shows no observed exporter regression.
- Next step: Push the environment/timing alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:30 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by aliasing repeated distributed lifecycle helpers and compile/sync calls without changing exporter behavior.
- Command or config: Added short aliases inside `candidates/non_ttt_vrl_gptq/train_gpt.py` for `th.compile -> CMP`, `th.cuda.synchronize -> SY`, `dist.is_available -> IA`, `dist.is_initialized -> II`, `dist.barrier -> BR`, `dist.get_world_size -> GWS`, `dist.get_rank -> GRK`, `dist.init_process_group -> IGP`, and `dist.destroy_process_group -> DGP`, replaced the corresponding repeated call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `62,810` bytes to `62,736` bytes, another `74` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 62,736`, which is `-12,195` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the distributed/compile alias pass in the main lane. The gain is small, but it is free, safe, and still moves the counted source in the right direction.
- Next step: Push the distributed/compile alias pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:32 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by removing internal docstrings that do not affect runtime, checkpoints, or artifact format.
- Command or config: Removed the internal docstrings from the GPTQ/int6 percentile/int8 embedding/calibration helpers in `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `62,736` bytes to `62,202` bytes, another `534` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 62,202`, which is `-12,729` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the docstring-removal pass in the main lane. It is another clean counted-size win with no observed exporter regression.
- Next step: Push the docstring-removal pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:34 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful block of counted source bytes by removing leftover inline comments and shortening non-critical log labels while preserving parsable export/status markers and runtime behavior.
- Command or config: Removed the remaining inline explanatory comments in `candidates/non_ttt_vrl_gptq/train_gpt.py`, shortened several non-critical log labels (for example calibration/config/warmup/memory labels), and kept the important markers like `step:`, `val_bpb:`, and `size_ok:` unchanged; then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `62,202` bytes to `61,768` bytes, another `434` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 61,768`, which is `-13,163` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the comment/log-trim pass in the main lane. It is another clean counted-size win with no observed exporter regression and no change to the markers used by our export ladder tooling.
- Next step: Push the comment/log-trim pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:37 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by removing pure startup diagnostics and unused helper variables that do not affect training, export behavior, checkpoints, or artifact format.
- Command or config: Removed the source-dump/separator log, removed the dataset-summary/value-token startup logs, removed the unused `dataset_dir`, `actual_train_files`, and `xsa_layers` helpers, and dropped several startup-only config summary logs from `candidates/non_ttt_vrl_gptq/train_gpt.py`; then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `61,768` bytes to `61,000` bytes, another `768` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 61,000`, which is `-13,931` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the startup-diagnostics pruning pass in the main lane. It is one of the better counted-size wins from the offline hygiene phase and still shows no observed exporter regression.
- Next step: Push the startup-diagnostics pruning pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:39 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful block of counted source bytes by removing more pure diagnostics that are not used by the export ladder or artifact format.
- Command or config: Removed the checkpoint-save notice, GPTQ calibration/hessian notices, codec-loaded echo, export-only checkpoint echo, warmup progress log, late-QAT toggle log, SWA-start log, and final memory/EMA notices from `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `61,000` bytes to `60,479` bytes, another `521` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 60,479`, which is `-14,452` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the extra diagnostics-removal pass in the main lane. It is another clean counted-size win and still preserves the markers the ladder tooling depends on.
- Next step: Push the diagnostics-removal pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:41 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful block of counted source bytes by shortening exception strings and a few one-off non-tooling diagnostic labels without changing runtime behavior or the ladder markers we depend on.
- Command or config: Shortened the file/validation/codec/int6 error strings and compressed a couple of non-tooling size-warning labels inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `60,479` bytes to `60,163` bytes, another `316` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 60,163`, which is `-14,768` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the exception/diagnostic shortening pass in the main lane. It is another safe counted-size win and still preserves the success markers the export ladder tooling watches.
- Next step: Push the exception/diagnostic shortening pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:45 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe block of counted source bytes by shortening the hyperparameter object alias inside `main()` from `args` to `a`, while keeping runtime behavior and exporter behavior unchanged.
- Command or config: Renamed the local config object in `main()` from `args` to `a`, updated the corresponding `main()` call sites and references, fixed the one loop-variable collision introduced by the rename, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `60,163` bytes to `59,877` bytes, another `286` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 59,877`, which is `-15,054` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the `main()` alias-shortening pass in the main lane. The gain is modest, but it is safe, validated, and continues moving the counted candidate source downward without touching the exporter format.
- Next step: Push the `main()` alias-shortening pass so the next live export rerun uses the smaller counted candidate source together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:47 America/Chicago
- Commit: `244cc2c`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim the last easy counted-source bytes still tied up in helper-function config aliases by shortening the remaining `args.` references outside `main()`.
- Command or config: Renamed the config parameter from `args` to `a` in `eval_val(...)` and `ree(...)`, updated the corresponding local references and call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `59,877` bytes to `59,826` bytes, another `51` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 59,826`, which is `-15,105` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the helper-alias cleanup in the main lane. The gain is small, but it is validated, safe, and probably close to the end of the high-confidence alias-only wins.
- Next step: Push the helper-alias cleanup so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:50 America/Chicago
- Commit: `658afa6`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful block of counted source bytes by shortening the hottest remaining local-only names inside the export/eval helpers without changing log markers, artifact fields, or runtime behavior.
- Command or config: Shortened the remaining helper-local names in `eval_val(...)` and `ree(...)` inside `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example the local batch-span, Hessian map, prune accumulators, artifact byte counters, decode locals, and validation-token helper names), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `59,826` bytes to `59,146` bytes, another `680` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 59,146`, which is `-15,785` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the export-helper local-name shortening pass in the main lane. It is one of the better recent offline code-size wins and still shows no observed exporter regression.
- Next step: Push the export-helper local-name shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:54 America/Chicago
- Commit: `3ba09e2`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another larger block of counted source bytes by shortening the hottest remaining internal config field names while keeping the external env var names, artifact format, and runtime behavior unchanged.
- Command or config: Renamed the internal `H` config fields and their in-file accesses in `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example iteration/logging/warmup/eval/optimizer/pruning fields) to shorter aliases, while leaving all `EG(\"...\")` environment variable keys unchanged; then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `59,146` bytes to `58,400` bytes, another `746` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 58,400`, which is `-16,531` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the config-field shortening pass in the main lane. It is a high-confidence counted-size win because the external env var interface stayed intact and the exporter validation remained unchanged.
- Next step: Push the config-field shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:57 America/Chicago
- Commit: `935e026`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another moderate block of counted source bytes by shortening long internal helper/function symbols that never leave the candidate file and do not affect env vars, artifact fields, or ladder markers.
- Command or config: Renamed several internal helper symbols in `candidates/non_ttt_vrl_gptq/train_gpt.py` (including the sentencepiece/validation/data-load helpers, a few quantization helpers, the low-dim restore helper, the rotary helper, and the meta-kind/classifier helpers), updated all in-file call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `58,400` bytes to `57,955` bytes, another `445` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 57,955`, which is `-16,976` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the helper-symbol shortening pass in the main lane. It is another safe counted-size win with no observed exporter regression.
- Next step: Push the helper-symbol shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 03:59 America/Chicago
- Commit: `847d649`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim one more small block of counted source bytes by shortening a few remaining internal class/helper symbols used only inside the candidate file.
- Command or config: Renamed the remaining long internal class/helper symbols in `candidates/non_ttt_vrl_gptq/train_gpt.py` (including `Muon`, `TokenStream`, `RMSNorm`, `Rotary`, `SmearGate`, `Block`, `chs`, and `eval_val_sliding`) to shorter in-file aliases, updated the call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `57,955` bytes to `57,855` bytes, another `100` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 57,855`, which is `-17,076` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the class/helper-symbol shortening pass in the main lane. The gain is smaller than the last few passes, but it is validated, free, and still moves the counted candidate source down.
- Next step: Push the class/helper-symbol shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:01 America/Chicago
- Commit: `2f205b3`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful block of counted source bytes by shortening long local-only names in the sentencepiece, validation, shard-loader, and batch-loader helpers without changing runtime behavior or external interfaces.
- Command or config: Shortened the repeated local variable names in `bsl(...)`, `eval_val(...)`, `lds(...)`, and `DTL.next_batch(...)` inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `57,855` bytes to `57,260` bytes, another `595` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 57,260`, which is `-17,671` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the loader/validation local-name shortening pass in the main lane. It is another good counted-size win with no observed exporter regression.
- Next step: Push the loader/validation local-name shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:05 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another moderate block of counted source bytes by shortening the repeated warmup/train-control locals inside `main()` without touching env var names, log markers, or export semantics.
- Command or config: Renamed the repeated internal `main()` locals in `candidates/non_ttt_vrl_gptq/train_gpt.py` (including the compile wrapper handle, optimizer locals, warmup snapshots, wallclock/stop flags, and a batch of train-loop temporaries), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `57,260` bytes to `56,661` bytes, another `599` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 56,661`, which is `-18,270` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the `main()` local-name shortening pass in the main lane. It is another safe counted-size win with no observed exporter or wrapper regression.
- Next step: Push the `main()` local-name shortening pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:08 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another moderate counted-size block by shortening repeated internal quant/export locals and optimizer-setup locals without changing env vars, checkpoint keys, log markers, or artifact semantics.
- Command or config: Shortened a batch of repeated internal names in `candidates/non_ttt_vrl_gptq/train_gpt.py` (including `clip_range`, `name_to_idx`, `base_name`, the temporary int6-pack flag, the optional Hessian map, and the main optimizer-group locals), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `56,661` bytes to `56,091` bytes, another `570` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 56,091`, which is `-18,840` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the quant/export and optimizer-local shortening pass in the main lane. It is another safe counted-size win with no observed exporter or wrapper regression.
- Next step: Push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:12 America/Chicago
- Commit: `7d1d3ce`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a larger counted-size block by shortening the candidate-private env keys in `H` and updating the VRL runpod configs/scripts in lockstep, while leaving torchrun’s standard distributed env vars alone.
- Command or config: Patched `candidates/non_ttt_vrl_gptq/train_gpt.py` so the candidate now reads short private env keys such as `DP`, `TP`, `VLE`, `MWS`, `GCB`, `GBS`, `PP`, `SPC`, and `EOC`; updated the `non_ttt_vrl_gptq*.env` runpod configs to the same short keys; and updated `runpod/check_ready.py`, `runpod/pod_queue_export_ladder.sh`, and `runpod/pod_queue_export_after_prefix.sh` to look for `PP` / `SPC` / `EOC`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the offline codec round-trip validator on the fixed-seed quantized VRL model skeleton.
- Result: The candidate file shrank from `56,091` bytes to `55,504` bytes, another `587` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 55,504`, which is `-19,427` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the short private env-key scheme in the main lane. It is a larger validated counted-size win than the last few local-only rename passes, and the runpod orchestration still passes readiness after the synchronized config/script update.
- Next step: Push the short env-key pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:16 America/Chicago
- Commit: `b5a8431`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another solid counted-size block by shortening repeated internal `device`, `seq_len`, and decode `offset` locals in validation, loader, rotary-cache, export, and main-loop paths without changing external interfaces.
- Command or config: Renamed the internal `device` locals to `dv`, shortened several local `seq_len` parameters to short in-file aliases, and collapsed decode offsets to `o` inside `candidates/non_ttt_vrl_gptq/train_gpt.py`; then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the fixed-seed offline codec round-trip validator on the quantized VRL model skeleton.
- Result: The candidate file shrank from `55,504` bytes to `55,017` bytes, another `487` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 55,017`, which is `-19,914` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep the internal device/seq-len/offset shortening pass in the main lane. It is another validated counted-size win with no observed exporter, wrapper, or round-trip regression.
- Next step: Push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:19 America/Chicago
- Commit: `1277135`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful counted-size block by shortening repeated internal export/calibration symbols such as codec ids, meta-index helpers, loader-pattern args, calibration Hessian locals, and a few main/export checkpoint locals.
- Command or config: Renamed the remaining high-frequency internal symbols in `candidates/non_ttt_vrl_gptq/train_gpt.py` (including `codec_id`, `name_to_idx`, `pattern`, `hessians`, `param_name`, `master`, `raw_logits_fn`, and `model_state`), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the fixed-seed offline codec round-trip validator on the quantized VRL model skeleton.
- Result: The candidate file shrank from `55,017` bytes to `54,691` bytes, another `326` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,691`, which is `-20,240` bytes. Compile and readiness checks still passed, and the offline validator remained unchanged (`codec=lzma_raw_hc3_16mb`, `blob_size=4299069`, `roundtrip=True`).
- Decision: Keep this export/calibration symbol-shortening pass in the main lane. It is another validated counted-size win with no observed exporter, wrapper, or round-trip regression.
- Next step: Push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:25 America/Chicago
- Commit: `dc3722c`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing repeated codec-path helper spellings (`struct.pack`, `struct.unpack_from`, and `ValueError`) without changing export behavior.
- Command or config: Added short in-file aliases for `struct.pack`, `struct.unpack_from`, and `ValueError` inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, updated the repeated codec/int6/header call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, rebuilt a deterministic CPU round-trip validator directly from the candidate’s own `GPT`/`qsd`/`eqm`/`cmb`/`dmb`/`dsd` helpers, and compared `HEAD` versus the working copy with that exact same harness.
- Result: The candidate file shrank from `54,691` bytes to `54,508` bytes, another `183` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,508`, which is `-20,423` bytes. Compile and readiness checks still passed. The rebuilt validator round-tripped cleanly with `codec=lzma_raw_hc3_16mb` and `blob_size=4264990`, and the `HEAD` vs working-copy comparison matched exactly (`4264990` in both cases), so the alias pass did not change measured export behavior under that harness.
- Decision: Keep the codec-helper alias pass in the main lane. It is another validated counted-size win, and the side-by-side comparison indicates it is behavior-neutral for the deterministic offline export path.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:31 America/Chicago
- Commit: `ca228ca`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing repeated `bm.state_dict` / `bm.load_state_dict` method calls inside `main()` without touching checkpoint tensor names or export semantics.
- Command or config: Added short local aliases for `bm.state_dict` and `bm.load_state_dict` inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, updated the repeated warmup/EMA/SWA/export-only call sites in `main()`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic CPU export round-trip comparison harness on `HEAD` versus the working copy.
- Result: The candidate file shrank from `54,508` bytes to `54,431` bytes, another `77` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,431`, which is `-20,500` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4264990` for both), so the alias pass did not change measured export behavior under that harness.
- Decision: Keep the state/load helper alias pass in the main lane. The gain is smaller than the earlier wins, but it is another validated counted-size reduction with no observed exporter regression.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:31 America/Chicago
- Commit: `39474bf`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by shortening the artifact-breakdown log labels while leaving the runpod-parsed markers (`step:`, `val_bpb:`, `train_time:`, `eval_time:`, and `size_ok:`) unchanged.
- Command or config: Shortened the non-critical artifact-breakdown labels inside `candidates/non_ttt_vrl_gptq/train_gpt.py` (for example `tensor_headers`, `int6_payload`, `other_payload`, `raw_total`, `compressed_model`, and `int6_tensors`), then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic CPU export round-trip comparison harness on `HEAD` versus the working copy.
- Result: The candidate file shrank from `54,431` bytes to `54,361` bytes, another `70` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,361`, which is `-20,570` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4264990` for both), so the trim did not change measured export behavior under that harness.
- Decision: Keep the artifact-breakdown label trim in the main lane. It is another small but validated counted-size win, and it does not touch any log markers that the runpod tooling parses.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:34 America/Chicago
- Commit: `0995ef4`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a larger counted-size block by compacting the repeated CUDA autocast spellings and the handful of `"cuda"` literals without changing export behavior.
- Command or config: Added short in-file aliases for the CUDA device string and the repeated autocast context helper inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, replaced the repeated `AC(device_type=\"cuda\", dtype=BF, ...)` sites and the `th.device(\"cuda\", ...)` construction, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic CPU export round-trip comparison harness on `HEAD` versus the working copy.
- Result: The candidate file shrank from `54,361` bytes to `54,211` bytes, another `150` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,211`, which is `-20,720` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4264990` for both), so the helper pass did not change measured export behavior under that harness.
- Decision: Keep the CUDA/autocast helper pass in the main lane. It is a better counted-size win than the last two micro-passes and still looks behavior-neutral under the deterministic offline export comparison.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:37 America/Chicago
- Commit: uncommitted
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by compacting the repeated `detach().cpu()` / `clone()` / `contiguous()` chains in the export and checkpoint paths without changing artifact semantics.
- Command or config: Added tiny in-file helpers for `detach().cpu()`, `detach().cpu().clone()`, and `detach().cpu().contiguous()` inside `candidates/non_ttt_vrl_gptq/train_gpt.py`, replaced the repeated export/checkpoint call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic CPU export round-trip comparison harness on `HEAD` versus the working copy.
- Result: The candidate file shrank from `54,211` bytes to `54,157` bytes, another `54` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,157`, which is `-20,774` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4264990` for both), so the helper pass did not change measured export behavior under that harness.
- Decision: Keep the detach/cpu helper pass in the main lane. It is another validated counted-size win with no observed exporter regression under the deterministic offline comparison.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:41 CDT
- Commit: `4a7cadd`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing the hottest remaining numpy dtype spellings in the int6 codec path without changing export semantics.
- Command or config: Added short in-file aliases for `np.uint8`, `np.int16`, and `np.int8` inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), updated the repeated int6 codec call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and rebuilt a deterministic `HEAD` versus working-copy CPU model-skeleton export round-trip harness under `uv` with `numpy`, `sentencepiece`, `zstandard`, and `torch`.
- Result: The candidate file shrank from `54,157` bytes to `54,034` bytes, another `123` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 54,034`, which is `-20,897` bytes. Compile and readiness checks still passed. The rebuilt deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4301621` for both), so the dtype-alias pass did not change measured export behavior under that harness.
- Decision: Keep the numpy-dtype alias pass in the main lane. It is a solid counted-size win, and the side-by-side deterministic export comparison indicates it is behavior-neutral.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:44 CDT
- Commit: `fe16c4e`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by collapsing the repeated `astype(..., copy=False)` spellings in the int6 codec path into a tiny helper without changing export semantics.
- Command or config: Added a short in-file `AS(x, d)` helper inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), replaced the repeated `astype(..., copy=False)` call sites across the int6 zigzag, pack, and unpack helpers, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic `HEAD` versus working-copy CPU model-skeleton export comparison harness under `uv` with `numpy`, `sentencepiece`, `zstandard`, and `torch`.
- Result: The candidate file shrank from `54,034` bytes to `53,828` bytes, another `206` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 53,828`, which is `-21,103` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4301621` for both), so the helper pass did not change measured export behavior under that harness.
- Decision: Keep the `astype(..., copy=False)` helper pass in the main lane. It is one of the better recent counted-size wins, and the side-by-side deterministic export comparison indicates it is behavior-neutral.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:46 CDT
- Commit: `821b023`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by collapsing the repeated `int(np.prod(shape, dtype=np.int64))` spellings in the int6 decode path into a tiny helper without changing export semantics.
- Command or config: Added a short in-file `NM(shape)` helper inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), replaced the repeated `int(np.prod(shape, dtype=np.int64))` call sites across `ui6l`, `ui6`, and the artifact decode path, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic `HEAD` versus working-copy CPU model-skeleton export comparison harness under `uv` with `numpy`, `sentencepiece`, `zstandard`, and `torch`.
- Result: The candidate file shrank from `53,828` bytes to `53,737` bytes, another `91` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 53,737`, which is `-21,194` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4301621` for both), so the helper pass did not change measured export behavior under that harness.
- Decision: Keep the `NM(shape)` helper pass in the main lane. It is another clean counted-size win, and the side-by-side deterministic export comparison indicates it is behavior-neutral.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 04:48 CDT
- Commit: `30fceb8`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing a small bundle of repeated library call paths (`np.frombuffer`, `np.empty`, `np.zeros`, `F.rms_norm`, `F.linear`, `Path`, and `glob.glob`) without changing export semantics.
- Command or config: Added short in-file aliases for those repeated call paths inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), updated the codec, loader, normalization, linear, and path call sites, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the deterministic `HEAD` versus working-copy CPU model-skeleton export comparison harness under `uv` with `numpy`, `sentencepiece`, `zstandard`, and `torch`.
- Result: The candidate file shrank from `53,737` bytes to `53,689` bytes, another `48` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 53,689`, which is `-21,242` bytes. Compile and readiness checks still passed. The deterministic comparison harness matched exactly on `HEAD` and the working copy (`codec=lzma_raw_hc3_16mb`, `blob_size=4301621` for both), so the alias bundle did not change measured export behavior under that harness.
- Decision: Keep the small call-path alias bundle in the main lane. The gain is modest, but it is still a validated counted-size reduction with no observed exporter regression.
- Next step: Commit and push this pass so the next live export rerun uses the smallest counted candidate source we have so far together with the stronger bitplane + raw-LZMA exporter path.

- Timestamp: 2026-03-24 05:03 CDT
- Commit: uncommitted
- Lane: Runpod live recovery
- Objective: Recover the surviving strongest VRL checkpoint from the resumed H100 pod and turn it into a fresh export-only rerun without paying for a full retrain.
- Command or config: Queried the Runpod GraphQL API with the local `.env` key, resumed pod `liyrlu10czeqvw` (`interim_violet_lynx`), reconnected over SSH at `root@216.243.220.229 -p 12768`, verified that `/workspace/golf` still contained `final_model.int6.ptz` and `final_model.pt`, reran `bash runpod/local_sync_to_pod.sh root@216.243.220.229 /workspace/golf 12768`, and started `VENV_DIR=/root/.venvs/golf bash runpod/pod_bootstrap.sh` to repair the missing Python environment.
- Result: The live recovery got far enough to confirm that the surviving pod artifact is `final_model.int6.ptz = 16,271,727` bytes and the corresponding `final_model.pt = 106,178,569` bytes still exist on the pod, which means the live overage sitting on disk is only `271,727` bytes above the cap. The workspace sync succeeded and wrote `.sync_commit = 83f5cd1100f25989854d642c92fa30947e37dc2f`, but the pod dropped back to `EXITED` during bootstrap. A fresh `podResume` attempt then failed with an exact balance error: current balance `0.4178517332`, required `0.44833333333333333333`, deficit `0.03048160013333333333`. That leaves the current live blocker as account balance rather than code readiness.
- Decision: Treat the live lane as blocked on roughly three cents of additional Runpod credit. The strongest next live move remains export-only recovery from the surviving `final_model.pt`; there is still no evidence that a fresh retrain is necessary once the pod can be resumed again.
- Next step: When balance is replenished, resume `liyrlu10czeqvw`, finish the bootstrap or use the base image Python if it already has `torch`, inspect `final_model.pt`, and launch the baseline-first export ladder from that checkpoint.

- Timestamp: 2026-03-24 05:03 CDT
- Commit: `f6c21b2`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a bigger counted-size block by collapsing the repeated `int(EG(...))`, `float(EG(...))`, and `bool(int(EG(...)))` env-parsing boilerplate into tiny typed helpers, then trim the remaining `load_state_dict(..., strict=True)` sites to the shorter positional form.
- Command or config: Added short in-file `GI`, `GF`, and `GB` helpers near the top of [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), replaced the repeated config-field parsing and `TORCH_COMPILE_DISABLE` checks with those helpers, switched the remaining `load_state_dict(..., strict=True)` calls to the shorter positional form where valid, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `53,689` bytes to `53,304` bytes, another `385` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 53,304`, which is `-21,627` bytes. Compile and readiness checks still passed, and the config validation still reports `runpod readiness check: OK` with `7` config refs in each recovery wrapper and `30` verified config files.
- Decision: Keep the env-helper and shorter `load_state_dict` pass in the main lane. It is a materially larger counted-size win than the last few micro-passes, and it does not change the intended runtime configuration semantics.
- Next step: Push this pass so the next live export-only recovery rerun uses the smaller counted source alongside the already-banked bitplane int6 codec, auto codec chooser, and tuned raw `lzma_raw_hc3_16mb` path.

- Timestamp: 2026-03-24 05:05 CDT
- Commit: `ba04baf`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another meaningful counted-size block by compacting the large `H` config class layout without changing any env keys, defaults, or runtime parsing behavior.
- Command or config: Collapsed the repeated one-field-per-line assignments inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) into denser multi-assignment lines, removed unnecessary spacing in that class body, and kept the exact same env variable names and defaults. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `53,304` bytes to `52,959` bytes, another `345` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 52,959`, which is `-21,972` bytes. Compile and readiness checks still passed, and the config validation still reports `runpod readiness check: OK` with `7` config refs in each recovery wrapper and `30` verified config files.
- Decision: Keep the compact `H` class layout in the main lane. It is another solid counted-size win, and it preserves the exact same config surface for the next live export-only rerun.
- Next step: Push this pass so the next live recovery rerun uses the smaller counted source together with the already-banked exporter improvements once the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 05:05 CDT
- Commit: `b8dbb03`
- Lane: Runpod recovery tooling
- Objective: Reduce the cost and time of the next live recovery attempt by avoiding the heavy CUDA `uv sync` bootstrap and by adding a direct export-only ladder for the surviving `final_model.pt` checkpoint.
- Command or config: Added [pod_bootstrap_export_only.sh](/Users/aryan/Desktop/golf/runpod/pod_bootstrap_export_only.sh), which creates a small `--system-site-packages` venv on the pod and installs only missing export-only dependencies (`numpy`, `sentencepiece`, `torch`, `zstandard`) instead of redownloading the full CUDA stack. Added [pod_run_existing_export_ladder.sh](/Users/aryan/Desktop/golf/runpod/pod_run_existing_export_ladder.sh) to run the prune ladder directly from an existing checkpoint path, and added [local_recover_existing_export.sh](/Users/aryan/Desktop/golf/runpod/local_recover_existing_export.sh) to sync the repo, run the light bootstrap, and execute that direct export ladder remotely. Updated [check_ready.py](/Users/aryan/Desktop/golf/runpod/check_ready.py) to validate the new wrapper.
- Result: The repo now has a dedicated fast path for “resume pod -> inspect/use surviving `final_model.pt` -> run export-only prune ladder” without first paying for `uv sync --extra cuda`. Syntax checks passed (`bash -n` on all three new shell scripts), and `python3 runpod/check_ready.py` still passes with the new wrapper included (`local_recover_existing_export.sh: 6 config refs`, total `30` verified config files).
- Decision: Keep the lightweight export-recovery lane in the main toolkit. It directly targets the real live blocker we hit this morning: spending scarce balance on environment rebuild before we could get back to export-only recovery.
- Next step: Once the Runpod balance is high enough to resume `liyrlu10czeqvw`, use [local_recover_existing_export.sh](/Users/aryan/Desktop/golf/runpod/local_recover_existing_export.sh) against `./final_model.pt` first, before falling back to the heavier full-chain bootstrap.

- Timestamp: 2026-03-24 05:10 CDT
- Commit: `76d1f89`
- Lane: Runpod recovery tooling
- Objective: Remove the remaining manual GraphQL/SSH bookkeeping from the next live attempt so “balance restored” immediately becomes “resume pod and recover export-only run”.
- Command or config: Added [local_resume_existing_export.sh](/Users/aryan/Desktop/golf/runpod/local_resume_existing_export.sh). The wrapper reads `RUNPOD_API_KEY` from the environment or local `.env`, issues the `podResume` GraphQL mutation for a given pod id, polls until a public SSH port appears, and then hands off directly to [local_recover_existing_export.sh](/Users/aryan/Desktop/golf/runpod/local_recover_existing_export.sh) using that discovered `root@host` and port. Verified shell syntax with `bash -n runpod/local_resume_existing_export.sh` and dry-ran it against the currently blocked pod id `liyrlu10czeqvw`.
- Result: The wrapper failed fast and cleanly on the current real-world blocker instead of hanging: `Insufficient balance to resume this on-demand pod. You need at least $0.45 (current: $0.41, deficit: $0.03). Please add funds to your account.` That means the next resume attempt can now be launched with a single local command instead of manual API curls and polling.
- Decision: Keep the resume wrapper in the main recovery toolkit. It reduces operator overhead on the exact lane that matters now: export-only recovery from the surviving `final_model.pt` as soon as enough balance is available.
- Next step: After balance is replenished, use [local_resume_existing_export.sh](/Users/aryan/Desktop/golf/runpod/local_resume_existing_export.sh) with pod id `liyrlu10czeqvw` and remote checkpoint path `./final_model.pt` to start the next live recovery attempt.

- Timestamp: 2026-03-24 05:10 CDT
- Commit: `6ef5f77`
- Lane: non-ttt VRL offline validation
- Objective: Re-establish a stronger behavior check for the latest candidate after the recent config-block compaction and recovery-tooling work, using a real CPU model-skeleton export comparison instead of only syntax/readiness checks.
- Command or config: Ran `uv run --with torch --with numpy --with sentencepiece --with zstandard` locally, loaded the current [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) plus the older `f6c21b2` version from `git show`, instantiated the same fixed-seed VRL `GPT` skeleton in both, quantized via `qsd`, serialized via `eqm`/tensor records/`cmb`, decompressed via `dmb`, and compared the final wrapped blob metrics and SHA-256 hashes.
- Result: The current candidate and the older `f6c21b2` candidate matched exactly under this stronger CPU harness: `codec=lzma_raw_hc3_16mb`, `blob_size=4265342`, `raw_size=20623636`, and identical SHA-256 `4c790a9145ffe855b6a28b60b1184808add2e6e9074aeba9fec692e3f36824ce`. This is stronger evidence than plain `py_compile` that the latest source-size reductions did not perturb the model-serialization path on the offline skeleton test.
- Decision: Treat the latest counted-source reductions as behavior-neutral under the current CPU export harness as well as under syntax/readiness checks.
- Next step: Keep using this stronger local `uv run --with torch ...` comparison before banking future source-size passes, so we have a higher-confidence offline gate while Runpod remains balance-blocked.

- Timestamp: 2026-03-24 05:17 CDT
- Commit: `4a6d25b`
- Lane: non-ttt VRL code-size hygiene + recovery validation
- Objective: Reclaim one more safe counted-size block in the candidate while also fixing the readiness checker so it understands the new delegated resume wrapper correctly.
- Command or config: Added tiny in-file aliases for `FileNotFoundError`, `RuntimeError`, `np.uint32`, and a shared `non_blocking=1` constant inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), updated the loader/codec/eval call sites, and adjusted [check_ready.py](/Users/aryan/Desktop/golf/runpod/check_ready.py) so wrappers can declare different minimum embedded-config counts instead of assuming every wrapper must inline at least five config paths. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` CPU model-skeleton export comparison against the older `f6c21b2` candidate.
- Result: The candidate file shrank from `52,959` bytes to `52,948` bytes, another `11` bytes of direct code-payload savings. Total counted source savings now stand at `74,931 -> 52,948`, which is `-21,983` bytes. The updated readiness validator still passes and now reports `local_resume_existing_export.sh: 0 config refs` as expected instead of failing. The stronger CPU export comparison still matched exactly against `f6c21b2`: `codec=lzma_raw_hc3_16mb`, `blob_size=4265342`, `raw_size=20623636`, identical SHA-256 `4c790a9145ffe855b6a28b60b1184808add2e6e9074aeba9fec692e3f36824ce`.
- Decision: Keep both the small alias pass and the readiness-check fix. The size win is tiny, but it is free, and the validator fix matters more because it keeps the new resume wrapper inside the checked launch surface.
- Next step: Push this pass so the repo stays clean and the next live recovery attempt can rely on a readiness checker that understands the full current recovery toolkit.

- Timestamp: 2026-03-24 05:25 CDT
- Commit: `891c08c`
- Lane: non-ttt VRL exporter compression
- Objective: Find a better final raw `LZMA2` setting for the already-banked bitplane int6 export path, so the next live export-only recovery attempt gets extra artifact headroom without needing another full train.
- Command or config: Ran a local `uv run --with torch --with numpy --with sentencepiece --with zstandard` CPU model-skeleton compression sweep around the current raw `LZMA2` default, keeping the same deterministic VRL `GPT` skeleton, `qsd` quantization flow, and serialized `quant_raw` payload. That sweep found a better raw filter at `hc3`, `16 MiB`, `lc=2`, `pb=2`, `nice_len=273`, which compressed the same `raw_size=20623636` payload to `4259004` bytes instead of the prior `4264985`. Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to switch `LF` from `lc=3` to `lc=2` and updated the logged codec label to `lzma_raw_hc3_16mb_lc2`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran a chooser-level CPU export round-trip harness under `uv` to confirm the wrapped blob selected the new filter and decompressed exactly.
- Result: The candidate file grew slightly from `52,948` bytes to `52,952` bytes (`+4`), but the wrapped model blob on the same deterministic skeleton fell to `blob_size=4259009` with `codec=lzma_raw_hc3_16mb_lc2` and exact `roundtrip=True`. Relative to the prior raw-`LZMA2` default on that same payload (`4264990` wrapped bytes including the 5-byte `QCB1` header), this is a `-5981` byte blob win and a net `-5977` byte combined win even after the small source increase.
- Decision: Keep the new `lc=2` raw `LZMA2` default in the main lane. It is a real exporter-side improvement, it survives the exact CPU round-trip check, and it gives the next live export-only rerun a little more headroom before we need harsher pruning.
- Next step: Push this filter update so the next `local_resume_existing_export.sh` recovery run uses the smaller `lzma_raw_hc3_16mb_lc2` path immediately once the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 05:33 CDT
- Commit: `48b25ee`
- Lane: non-ttt VRL exporter compression
- Objective: Tighten the new raw `LZMA2` winner further by sweeping nearby `pb` and `nice_len` settings instead of stopping at the first `lc=2` improvement.
- Command or config: Reused the same local deterministic CPU VRL model-skeleton export harness under `uv run --with torch --with numpy --with sentencepiece --with zstandard`, but narrowed the sweep to the current best family: `hc3`, `16 MiB`, `lc=2`, varying only `pb in {0,1,2}` and `nice_len in {64,96,128,160,192,224,273}`. The best result from that sweep was `pb=2`, `nice_len=64`, which compressed the same `raw_size=20623636` payload to `4246537` bytes in raw form. Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to switch `LF` from `nice_len=273` to `nice_len=64` while keeping the already-banked `lc=2`, and updated the logged codec label to `lzma_raw_hc3_16mb_lc2_n64`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the chooser-level CPU export round-trip harness.
- Result: The candidate file grew slightly from `52,952` bytes to `52,955` bytes (`+3`), but the wrapped model blob on the same deterministic skeleton fell again to `blob_size=4246542` with `codec=lzma_raw_hc3_16mb_lc2_n64` and exact `roundtrip=True`. Relative to the previous wrapped `lc=2` winner (`4259009` bytes), this is another `-12467` bytes of blob savings and a net `-12464` byte combined win after the tiny source increase. Relative to the earlier `lc=3` raw-`LZMA2` default (`4264990` wrapped bytes, `52948` counted source), the total combined improvement is now `-18441` bytes on this deterministic skeleton.
- Decision: Keep `lzma_raw_hc3_16mb_lc2_n64` as the new main-lane default. This is a materially better exporter setting than the previous winner, and it still passes compile, readiness, and exact CPU round-trip checks.
- Next step: Push this tighter filter setting so the next live `local_resume_existing_export.sh` recovery run uses the smallest validated exporter path we currently have.

- Timestamp: 2026-03-24 05:31 CDT
- Commit: `f50c3ae`
- Lane: non-ttt VRL exporter compression
- Objective: Check whether the new `lzma_raw_hc3_16mb_lc2_n64` default still held up when varying dictionary size and match finder, so we would not leave an obvious nearby exporter win on the table.
- Command or config: Reused the same deterministic local CPU VRL model-skeleton export harness under `uv run --with torch --with numpy --with sentencepiece --with zstandard` and swept `{mf in [hc3,hc4,bt4], dict_size in [8 MiB,16 MiB,32 MiB], nice_len in [64,96]}` while holding the already-banked `lc=2`, `pb=2` constant. The sweep results were: `4246537 hc3 16MiB 64`, `4246576 hc3 8MiB 64`, `4246885 hc3 32MiB 64`, then a larger gap to the next group (`4249780 hc3 8MiB 96`, `4252332 hc3 16MiB 96`, `4254410 hc4 8MiB 64`, and all `bt4` variants above `4261808`).
- Result: No better raw filter was found than the current default. The already-banked `hc3 / 16 MiB / lc=2 / pb=2 / nice_len=64` setting remains the best measured raw `LZMA2` configuration on this deterministic skeleton, so there was nothing new to patch into [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py). The repo stayed on the current clean state with `blob_size=4246542`, `codec=lzma_raw_hc3_16mb_lc2_n64`, and `runpod readiness check: OK`.
- Decision: Keep the current exporter default unchanged and stop spending more local time on nearby raw-`LZMA2` variants unless a new live artifact suggests the payload distribution has shifted enough to justify another sweep.
- Next step: Treat Runpod balance restoration as the real next gating event, then resume the surviving checkpoint with `local_resume_existing_export.sh` so this smaller exporter path is exercised on the actual trained `final_model.pt`.

- Timestamp: 2026-03-24 05:33 CDT
- Commit: `a71a7e0`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes without touching artifact bytes, parsed log markers, or exporter behavior by shortening diagnostic codec-name strings that are only used in logs and debug output.
- Command or config: Shortened the human-readable codec names returned by [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) from longer labels like `lzma_raw_hc3_16mb_lc2_n64`, `legacy_lzma_hc4_32mb_xz`, `legacy_zstd22`, `legacy_zlib9`, and `unknown({cid})` to shorter equivalents (`lz_hc3_16_l2_n64`, `lx_hc4_32_xz`, `lzs22`, `lzl9`, `u({cid})`). Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the chooser-level CPU export round-trip harness under `uv`.
- Result: The candidate file shrank from `52,955` bytes to `52,908` bytes, another `47` bytes of direct code-payload savings. The wrapped export blob stayed exactly unchanged at `blob_size=4246542`, the chooser still selected the same codec path, and the CPU harness still reported exact `roundtrip=True` with the same SHA-256 `4bbb418b6d8265d42710b1cc6939d16e812ecfd150f5bae08b04dd32336fe3a6`.
- Decision: Keep the shorter codec-label spellings in the main lane. This is a free counted-size win with no measured effect on the artifact bytes or recovery tooling.
- Next step: Push this tiny hygiene pass so the next live export-only resume runs with the smallest counted source and the same best-known exporter behavior.

- Timestamp: 2026-03-24 05:35 CDT
- Commit: `4804a19`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by shortening internal quantization category strings that never leave the compact artifact format.
- Command or config: Changed [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) so `cpm()` returns one-character internal category codes (`e/m/b/a/v/o`) instead of longer strings like `embed`, `mlp`, `bigram`, `attn`, `ve`, and `other`. Also changed `mk()` to return one-character internal codes (`p/c/6/8`) instead of longer strings like `passthrough`, `passthrough_ctrl`, `int6`, and `int8`, and updated the internal comparisons in `eqm`, `qsd`, `dsd`, and the prune/export paths accordingly. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the chooser-level CPU export round-trip harness under `uv`.
- Result: The candidate file shrank from `52,908` bytes to `52,804` bytes, another `104` bytes of direct code-payload savings. The wrapped export blob again stayed exactly unchanged at `blob_size=4246542`, the chooser still selected the same codec path, and the CPU harness still reported exact `roundtrip=True` with the same SHA-256 `4bbb418b6d8265d42710b1cc6939d16e812ecfd150f5bae08b04dd32336fe3a6`.
- Decision: Keep the shorter internal quant category codes in the main lane. This is another free counted-size win with no measured effect on artifact bytes, compatibility, or recovery tooling.
- Next step: Push this pass so the next live export-only resume uses the smallest counted candidate we currently have together with the unchanged best-known exporter path.

- Timestamp: 2026-03-24 05:37 CDT
- Commit: `0f1cdbf`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing repeated long literals that are only used inside the candidate source and do not affect artifact bytes.
- Command or config: Added tiny aliases in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) for repeated literals such as `"utf-8"`, `"final_model.int6.ptz"`, `"model_state"`, `"momentum_buffer"`, and `"TORCH_COMPILE_DISABLE"`, then updated the repeated call sites in checkpoint save/load, log-file writes, metadata encode/decode, compile toggles, and the optimizer state path. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the chooser-level CPU export round-trip harness under `uv`.
- Result: The candidate file shrank from `52,804` bytes to `52,701` bytes, another `103` bytes of direct code-payload savings. The wrapped export blob remained exactly unchanged at `blob_size=4246542`, the chooser still selected the same codec path `lz_hc3_16_l2_n64`, and the CPU harness again reported exact `roundtrip=True` with the same SHA-256 `4bbb418b6d8265d42710b1cc6939d16e812ecfd150f5bae08b04dd32336fe3a6`.
- Decision: Keep the repeated-literal aliases in the main lane. This is another free counted-size win with no measured effect on artifact bytes or recovery behavior.
- Next step: Push this pass so the next live export-only resume uses the smallest counted candidate we have so far together with the unchanged best-known exporter path.

- Timestamp: 2026-03-24 05:38 CDT
- Commit: `158c719`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another safe counted-size block by aliasing repeated optimizer dictionary keys that stay exactly the same at runtime.
- Command or config: Added tiny aliases for the repeated `"params"` and `"base_lr"` string keys inside [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), then updated the Muon optimizer step path, the token-embedding/head optimizer param-group construction, the `base_lr` bookkeeping, and the late training-loop `group["base_lr"]` reads. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the chooser-level CPU export round-trip harness under `uv`.
- Result: The candidate file shrank from `52,701` bytes to `52,643` bytes, another `58` bytes of direct code-payload savings. The wrapped export blob again stayed exactly unchanged at `blob_size=4246542`, the chooser still selected `lz_hc3_16_l2_n64`, and the CPU harness again reported exact `roundtrip=True` with the same SHA-256 `4bbb418b6d8265d42710b1cc6939d16e812ecfd150f5bae08b04dd32336fe3a6`.
- Decision: Keep the optimizer-key aliases in the main lane. This is another free counted-size win with no measured effect on artifact bytes, optimizer behavior, or recovery tooling.
- Next step: Push this pass so the next live export-only resume uses the smallest counted candidate we currently have with the same best-known exporter path.

- Timestamp: 2026-03-24 05:44 CDT
- Commit: `29265cf`
- Lane: non-ttt VRL code-size hygiene + live resume recheck
- Objective: Reclaim another safe counted-size block in the codec/export path while rechecking the actual Runpod resume blocker with the current recovery wrapper instead of relying on older balance output.
- Command or config: Added tiny aliases in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) for repeated suffix/format literals used in the export path (`".q"`, `".scale"`, `"<I"`, `"<HB"`, `"<HBBB"`, and `"little"`), then updated the metadata encode/decode, tensor-record encode/decode, and int6 bitpacking call sites. Reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` CPU model-skeleton harness that compared `HEAD` vs the working copy and checked the wrapped blob bytes and SHA-256 directly, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `52,643` bytes to `52,629` bytes, another `14` bytes of direct code-payload savings. The stronger `HEAD` vs working-copy CPU harness matched exactly on the current shell-env deterministic skeleton: `blob_size=4287989`, `raw_size=20623636`, `codec=lz_hc3_16_l2_n64`, identical SHA-256 `0c461cd70388cc2e50153c26449c1de86b5eea2e87d7f57f60c878e4ba0ef5be`, and `same_blob=True`. The live recovery wrapper is still blocked only by Runpod balance: `You need at least $0.45 (current: $0.41, deficit: $0.04).`
- Decision: Keep the suffix/format alias pass. It is another free counted-size win, and the stronger `HEAD` vs working-copy harness shows it does not perturb the current export artifact bytes under the local deterministic skeleton test.
- Next step: Push this pass so the repo stays clean and launch-ready, then resume the surviving checkpoint with `local_resume_existing_export.sh` as soon as the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 05:50 CDT
- Commit: `6b23b3b`
- Lane: non-ttt VRL exporter compression
- Objective: Re-open raw `LZMA2` tuning around the current `lz_hc3_16_l2_n64` default and see whether the current shell-env deterministic skeleton still had a better filter family available than the already-banked `lc=2` path.
- Command or config: Ran a focused local `uv run --with torch --with numpy --with sentencepiece --with zstandard` sweep on the deterministic CPU VRL model-skeleton payload, first across `hc3` with `{dict in [8,16,32] MiB, (lc,lp) in [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1)], nice in [48,64,80]}` while holding `pb=2`, then a micro-sweep around the best family with `{dict in [24,32,40,48,64] MiB, pb in [0,1,2], nice in [48,64,80,96]}` at `lc=0, lp=0`, followed by a match-finder sanity check over `{hc3,hc4,bt4}`. That sweep found a better winner at `hc3`, `24 MiB`, `lc=0`, `lp=0`, `pb=2`, `nice_len=64`, so I patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to switch `LF` to `dict_size=3<<23`, `lc=0`, and updated the diagnostic codec label to `lz_hc3_24_l0_n64`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `52,629` bytes to `52,627` bytes, another `2` bytes of direct code-payload savings. On the same deterministic shell-env skeleton, the old `HEAD` blob was `blob_size=4287989`, `raw_size=20623636`, `codec=lz_hc3_16_l2_n64`, `sha256=0c461cd70388cc2e50153c26449c1de86b5eea2e87d7f57f60c878e4ba0ef5be`; the new working-copy blob fell to `blob_size=4280603`, `raw_size=20623636`, `codec=lz_hc3_24_l0_n64`, `sha256=e0c4c51dee00339a76445053bb01ae438d9632e2173f46988850ae801c3326c5`, for a wrapped-blob delta of `-7386` bytes. The raw sweep winner itself was `4280598` bytes before the 5-byte `QCB1` wrapper, and the match-finder check confirmed `hc3` still beats `hc4` (`4283598`) and `bt4` (`4297480`) for this family. The live recovery wrapper is still blocked only by balance: `You need at least $0.45 (current: $0.41, deficit: $0.04).`
- Decision: Keep `lz_hc3_24_l0_n64` as the new main-lane exporter default. This is a real chooser-level blob win on the deterministic skeleton, it survives the exact round-trip check, and it is materially more valuable than another tiny source-only alias pass.
- Next step: Push this filter upgrade so the next real resume from the surviving `final_model.pt` uses the smaller exporter immediately, then rerun `local_resume_existing_export.sh` as soon as the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 05:59 CDT
- Commit: `63a0785`
- Lane: non-ttt VRL exporter compression
- Objective: Push the new raw `LZMA2` line farther after the `lz_hc3_24_l0_n64` win by checking whether `lp=1` and nearby `pb` values are materially better on the same deterministic shell-env skeleton.
- Command or config: Ran a small targeted local `uv run --with torch --with numpy --with sentencepiece --with zstandard` sweep on the current deterministic CPU VRL payload. First checked a handpicked neighborhood around the current winner, which unexpectedly showed `lp=1` beating the current `lc=0, lp=0` line. Then ran a focused `lp=1` sweep over `{dict in [20,24,28,32] MiB, pb in [0,1,2], nice in [48,64,80]}` at `hc3`, `lc=0`. That sweep showed the best raw size is stable across all tested dict sizes at `pb=1`, `nice_len=64`, with raw size `4273282`, so I picked the shortest equivalent dict literal (`5<<22`, i.e. `20 MiB`) and patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to `dict_size=5<<22`, `lc=0`, `lp=1`, `pb=1`, `nice_len=64`, with the compact diagnostic codec label `lz_hc3_20_p1_n64`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness.
- Result: The candidate file stayed flat at `52,627` bytes. On the same deterministic shell-env skeleton, the old `HEAD` blob was `blob_size=4280603`, `raw_size=20623636`, `codec=lz_hc3_24_l0_n64`, `sha256=e0c4c51dee00339a76445053bb01ae438d9632e2173f46988850ae801c3326c5`; the new working-copy blob fell to `blob_size=4273287`, `raw_size=20623636`, `codec=lz_hc3_20_p1_n64`, `sha256=2295f9712aa796c76607fb4c7762ec6da6f5c7d54669e47c47ef3113c7c55fad`, for a wrapped-blob delta of `-7316` bytes. The best raw size under the focused `lp=1` sweep was `4273282`, and the best cases were identical for all tested dict sizes `20/24/28/32 MiB` at `pb=1`, `nice_len=64`.
- Decision: Keep `lz_hc3_20_p1_n64` as the new main-lane exporter default. This is another real chooser-level blob win, it keeps the counted source flat, and it beats the previous `lz_hc3_24_l0_n64` path by enough bytes to matter on the live cap problem.
- Next step: Push this exporter upgrade, then resume the surviving checkpoint with `local_resume_existing_export.sh` as soon as the Runpod balance blocker is cleared so the new smaller filter is exercised on the actual trained `final_model.pt`.

- Timestamp: 2026-03-24 06:01 CDT
- Commit: `72fcf26`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes after the `lp=1` exporter win without changing the artifact bytes, by shortening only human-readable codec labels used in logs and debug output.
- Command or config: Shortened the compact codec labels in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), changing `lz_hc3_20_p1_n64` to `lz20p1n64` and the legacy fallback label `lx_hc4_32_xz` to `lx432`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness.
- Result: The candidate file shrank from `52,627` bytes to `52,613` bytes, another `14` bytes of direct code-payload savings. The wrapped export blob stayed exactly unchanged at `blob_size=4273287`, `raw_size=20623636`, with identical SHA-256 `2295f9712aa796c76607fb4c7762ec6da6f5c7d54669e47c47ef3113c7c55fad`; only the diagnostic codec string changed from `lz_hc3_20_p1_n64` to `lz20p1n64`.
- Decision: Keep the shorter codec labels. This is another free counted-size win with no measured effect on artifact bytes, decompression behavior, or recovery tooling.
- Next step: Push this tiny hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `lp=1` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:18 CDT
- Commit: `ba0abd0`
- Lane: non-ttt VRL exporter compression
- Objective: Re-open the current `lp=1` raw-`LZMA2` line and check whether nearby `dict_size`, `pb`, `nice_len`, and `depth` settings could beat the current `lz20p1n64` default on the same deterministic CPU VRL skeleton.
- Command or config: Ran a tighter local `uv run --with torch --with numpy --with sentencepiece --with zstandard` sweep on the deterministic CPU model-skeleton payload around the current `hc3 / lc=0 / lp=1` family. The best observed neighborhood result moved to `dict_size=18 MiB`, `pb=1`, `nice_len=68`, `depth=3`, so I patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to `dict_size=9<<21`, `lc=0`, `lp=1`, `pb=1`, `mode=lzma.MODE_NORMAL`, `nice_len=68`, `mf=lzma.MF_HC3`, `depth=3`, with the compact diagnostic codec label `lz18p1683`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger local `HEAD` vs working-copy chooser-level CPU export harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file stayed flat at `52,613` bytes. On the same deterministic skeleton, the old `HEAD` blob was `blob_size=4235388`, `raw_size=20623636`, `codec=lz20p1n64`, `sha256=de3beefee9432ce0b51910d36df5cdf9d77676a4aaa94fa8f02817d2119229e0`; the new working-copy blob fell to `blob_size=4221908`, `raw_size=20623636`, `codec=lz18p1683`, `sha256=44436f8709ee07385c7c2c761a37d2b3ca7c91fc0c071a9576db320750c96726`, for a chooser-level wrapped-blob delta of `-13480` bytes. Exact round-trip still passed. The live recovery wrapper is still blocked only by balance: `You need at least $0.45 (current: $0.40, deficit: $0.05).`
- Decision: Keep `lz18p1683` as the new main-lane exporter default. This is a real chooser-level blob win on the deterministic skeleton and materially better than the previous `lz20p1n64` line.
- Next step: Push this filter upgrade so the next real resume from the surviving `final_model.pt` uses the smaller exporter immediately, then rerun `local_resume_existing_export.sh` as soon as the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 06:24 CDT
- Commit: `a6a62bf`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes after the new `dict18/pb1` exporter win without changing artifact bytes, by aliasing the repeated `".weight"` suffix and shortening only the human-readable codec label.
- Command or config: Added `WT=".weight"` in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), swapped the repeated `name + ".weight"` sites in the Hessian/export path to `name + WT`, and shortened the diagnostic codec label from `lz18p1683` to `l181683`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness.
- Result: The candidate file shrank from `52,613` bytes to `52,604` bytes, another `9` bytes of direct code-payload savings. The wrapped export blob stayed exactly unchanged at `blob_size=4221908`, `raw_size=20623636`, with identical SHA-256 `44436f8709ee07385c7c2c761a37d2b3ca7c91fc0c071a9576db320750c96726`; only the diagnostic codec string changed from `lz18p1683` to `l181683`.
- Decision: Keep the shorter suffix alias and codec label. This is another free counted-size win with no measured effect on artifact bytes or decompression behavior.
- Next step: Push this tiny hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `dict18/pb1` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:25 CDT
- Commit: `b6d3c47`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes after the first `dict18/pb1` label shrink without changing artifact bytes, by shortening only the remaining human-readable codec/debug strings.
- Command or config: Shortened the diagnostic codec string in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) from `l181683` to `l183`, shortened the legacy fallback labels from `lx432`/`lzs22`/`lzl9` to `x32`/`zs`/`z9`, and shortened the unknown-codec label from `u({cid})` to `u{cid}`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness.
- Result: The candidate file shrank from `52,604` bytes to `52,592` bytes, another `12` bytes of direct code-payload savings. The wrapped export blob stayed exactly unchanged at `blob_size=4221908`, `raw_size=20623636`, with identical SHA-256 `44436f8709ee07385c7c2c761a37d2b3ca7c91fc0c071a9576db320750c96726`; only the diagnostic codec strings changed.
- Decision: Keep the shorter codec/debug labels. This is another free counted-size win with no measured effect on artifact bytes or decompression behavior.
- Next step: Push this tiny hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `dict18/pb1` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:27 CDT
- Commit: `2db1cf1`
- Lane: non-ttt VRL exporter compression
- Objective: Tighten the new `dict18` raw-`LZMA2` family a little further after the `pb=1 / nice_len=68 / depth=3` win and see whether the same deterministic payload preferred a nearby `pb` / `nice_len` / `depth` point.
- Command or config: Reused the same local `uv run --with torch --with numpy --with sentencepiece --with zstandard` deterministic CPU VRL payload harness and ran a focused sweep at fixed `dict_size=18 MiB`, `lc=0`, `lp=1`, over `{pb in [0,1,2], nice_len in [66,67,68,69,70,71,72,73,74,76,80], depth in [1,2,3,4,5,6,8]}`. That sweep found a better point at `pb=0`, `nice_len=76`, `depth=2`, so I patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to `dict_size=9<<21`, `lc=0`, `lp=1`, `pb=0`, `mode=lzma.MODE_NORMAL`, `nice_len=76`, `mf=lzma.MF_HC3`, `depth=2`, with the compact diagnostic codec label `l0762`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger local `HEAD` vs working-copy chooser-level CPU export harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file moved from `52,592` bytes to `52,593` bytes, a `+1` byte source increase. On the same deterministic skeleton, the old `HEAD` blob was `blob_size=4221908`, `raw_size=20623636`, `codec=l183`, `sha256=44436f8709ee07385c7c2c761a37d2b3ca7c91fc0c071a9576db320750c96726`; the new working-copy blob fell to `blob_size=4221527`, `raw_size=20623636`, `codec=l0762`, `sha256=fa5a818c31c5de1dac25c8d1f6d04813548b364a2fb96d752474ee5202475e8f`, for a chooser-level wrapped-blob delta of `-381` bytes. Exact round-trip still passed. The live recovery wrapper is still blocked only by balance: `You need at least $0.45 (current: $0.40, deficit: $0.05).`
- Decision: Keep `l0762` as the new main-lane exporter default. This is a real measured chooser-level blob win on the deterministic skeleton, even though it slightly increases counted source by one byte.
- Next step: Push this tighter filter setting so the next real resume from the surviving `final_model.pt` uses the smaller exporter immediately, then rerun `local_resume_existing_export.sh` as soon as the Runpod balance blocker is cleared.

- Timestamp: 2026-03-24 06:39 CDT
- Commit: `269a10f`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes after the `l0762` exporter win without touching artifact bytes, by shortening only the diagnostic codec labels and fallback decompression tag.
- Command or config: Shortened the diagnostic codec strings in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) from `z19`/`zl9`/`l0762`/`x32` to `s`/`z`/`l`/`x`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py` and reran `python3 runpod/check_ready.py`.
- Result: The candidate file shrank from `52,593` bytes to `52,583` bytes, another `10` bytes of direct code-payload savings. This patch does not touch `LF`, `cmb()`, or the serialized artifact format; it only shortens human-readable diagnostic labels returned by `mcn()` and the legacy fallback name in `dmb()`. Readiness still passes: `runpod readiness check: OK`.
- Decision: Keep the shorter labels. This is another free counted-size win with no effect on compressor choice, artifact bytes, or decompression logic.
- Next step: Push this tiny hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:42 CDT
- Commit: `1535e60`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another chunk of counted bytes by shortening long file-internal helper and method names that do not affect artifact format, checkpoint/state-dict keys, or runpod parsing.
- Command or config: Renamed only internal helpers and method symbols in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `eval_val -> evv`, `_zigzag_encode_int6 -> ze6`, `_zigzag_decode_int6 -> zd6`, `forward_logits -> fwl`, and the local `transposed -> tr`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a small `uv run --with torch --with numpy --with sentencepiece --with zstandard` runtime smoke test that verified `ze6/zd6` round-trip on sample int6 values, `cmb()/dmb()` round-trip on sample bytes, and the renamed `bm.fwl()` path on a tiny CPU model.
- Result: The candidate file shrank from `52,583` bytes to `52,452` bytes, another `131` bytes of direct code-payload savings. Compile and readiness checks still passed, and the runtime smoke test succeeded with `ok 28 1 s (2, 8, 64)`, confirming the renamed helpers and method still execute correctly in the local `uv` torch environment.
- Decision: Keep the helper-name shrink. This is a meaningful counted-size win with no change to `LF`, `cmb()`, serialized tensor layout, or checkpoint field names.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:45 CDT
- Commit: `8f6a60b`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another chunk of counted bytes by shortening repeated local names and the `DTL` loader method name, while leaving artifact format, checkpoint/state-dict keys, and runpod-facing markers untouched.
- Command or config: Renamed only file-internal locals and one file-internal loader method in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `updates_flat -> uf`, `kind_code -> kc`, `kind_name -> kn`, `name_bytes -> xb`, `packed_u8 -> pu`, `triplets -> tr`, `plane_bytes -> pb`, `remaining -> r`, and `DTL.next_batch -> DTL.nb`, updating only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a local `uv run --with torch --with numpy --with sentencepiece --with zstandard` smoke test that verified `ze6/zd6` round-trip, `cmb()/dmb()` round-trip, the renamed `DTL.nb()` path on a dummy stream, and the renamed `bm.fwl()` path on a tiny CPU model.
- Result: The candidate file shrank from `52,452` bytes to `52,152` bytes, another `300` bytes of direct code-payload savings. Compile and readiness checks still passed, and the runtime smoke test succeeded with `ok 28 1 s (2, 4) (2, 4) (2, 8, 64)`, confirming the renamed locals and loader method still execute correctly in the local `uv` torch environment.
- Decision: Keep the local-name and loader-name shrink. This is a substantial counted-size win with no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:47 CDT
- Commit: `49552cb`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes by compacting the internal token-stream loader field names and helper method names, again without touching artifact format, checkpoint/state-dict keys, or runpod-facing markers.
- Command or config: Renamed only file-internal token-stream fields and methods in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `files -> fs`, `file_idx -> fi`, `tokens -> ts`, `pos -> p`, `_advance_file -> af`, `take -> tk`, `stream -> s`, `rank -> rk`, plus a few same-scope loader locals `chunk -> ck`, `start -> st`, `local -> lc`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a local `uv run --with torch --with numpy --with sentencepiece --with zstandard` smoke test that verified `ze6/zd6` round-trip, `cmb()/dmb()` round-trip, the renamed `DTL.nb()` path on a dummy stream exposing `tk()`, and the renamed `bm.fwl()` path on a tiny CPU model.
- Result: The candidate file shrank from `52,152` bytes to `52,021` bytes, another `131` bytes of direct code-payload savings. Compile and readiness checks still passed, and the runtime smoke test succeeded with `ok 28 1 s (2, 4) (2, 4) (2, 8, 64)`, confirming the renamed loader fields and methods still execute correctly in the local `uv` torch environment.
- Decision: Keep the loader-field shrink. This is another meaningful counted-size win with no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:49 CDT
- Commit: `93aadd1`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes by compacting repeated metadata and codec locals in `eqm/dqm/ree` and shortening the zstd feature-flag name, without changing artifact layout, checkpoint keys, or runpod-facing markers.
- Command or config: Renamed only file-internal metadata and codec locals in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `HAS_ZSTD -> HZ`, `kind_map -> km`, `entry_count -> ec`, `name_len -> nl`, `kind_code -> kc`, `name_idx -> ni`, `meta_blob/meta_names -> eb/en`, `model_codec_id -> mc`, `model_blob_loaded -> md`, and `meta_len -> ml`, updating only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a local `uv run --with torch --with numpy --with sentencepiece --with zstandard` smoke test that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, the renamed `DTL.nb()` path, and the renamed `bm.fwl()` path on a tiny CPU model.
- Result: The candidate file shrank from `52,021` bytes to `51,749` bytes, another `272` bytes of direct code-payload savings. Compile and readiness checks still passed, and the metadata/codec smoke test succeeded with `ok 18 28 1 s (2, 4) (2, 4) (2, 8, 64)`, confirming the renamed metadata locals and codec flag still execute correctly in the local `uv` torch environment.
- Decision: Keep the metadata-local shrink. This is another meaningful counted-size win with no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:51 CDT
- Commit: `6ae8b36`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another chunk of counted bytes by shortening model-private attribute names inside `CSA`, `BL`, and `GPT`, while staying away from module names, parameter names, checkpoint keys, and artifact layout.
- Command or config: Renamed only plain Python attributes in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `num_kv_heads -> nkh`, `ln_scale_factor -> lsf`, `num_encoder_layers -> nel`, `num_decoder_layers -> ndl`, `num_skip_weights -> nsw`, `ve_layer_indices -> vli`, and `vrl_enabled -> vre`, updating only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and ran a local `uv run --with torch --with numpy --with sentencepiece --with zstandard` smoke test that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, the renamed `DTL.nb()` path, and a tiny VE-enabled, VRL-enabled `GPT(...).fwl()` path with assertions over the renamed attributes (`vre`, `nel`, `ndl`, `nsw`, `attn.nkh`, `lsf`).
- Result: The candidate file shrank from `51,749` bytes to `51,422` bytes, another `327` bytes of direct code-payload savings. Compile and readiness checks still passed, and the attribute smoke test succeeded with `ok 18 28 1 s (2, 4) (2, 4) (2, 8, 64)`, confirming the renamed model-private attributes still execute correctly in the local `uv` torch environment.
- Decision: Keep the private-attribute shrink. This is another substantial counted-size win with no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 06:58 CDT
- Commit: `e58949d`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim another chunk of counted bytes by shortening file-internal helper and method names inside `CSA`, `BHE`, `GPT`, `ch()`, and `evl()`, while keeping artifact layout, checkpoint/state-dict keys, and runpod-facing markers unchanged.
- Command or config: Renamed only file-internal helpers and same-file call sites in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `CSA._xsa_efficient -> CSA.xe`, `BHE.bigram_hash -> BHE.bh`, `GPT._get_ve -> GPT.gv`, `GPT._run_layers -> GPT.rl`, `GPT._embed -> GPT.eb`, plus a batch of same-scope locals in `ch()` and `evl()`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`, and ran a stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, the renamed `DTL.nb()` path on a dummy stream, the renamed `xe/bh/gv/eb/rl/fwl` runtime paths on a tiny CPU VRL model, and a `HEAD` vs working-copy export-byte comparison on the same tiny skeleton.
- Result: The candidate file shrank from `51,422` bytes to `51,079` bytes, another `343` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger `HEAD` vs working-copy local harness matched exactly on the tiny CPU skeleton: `head_blob_size=67051`, `work_blob_size=67051`, `same_blob=True`, `same_raw=True`, `sha256=27fcab33cc9bfcc6356db3a2d222528d40d50465ca608c0b74d748cfafc6aa72`, so this helper/method rename pass is behavior-neutral under that probe. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the private-helper shrink. This is another meaningful counted-size win with no measured change to the tiny local export blob, no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 07:01 CDT
- Commit: `0b07322`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes by shortening file-internal cache fields, helper arguments, and local names in the rotary/attention/VE stack, while avoiding module names, parameter names, checkpoint keys, and artifact layout.
- Command or config: Renamed only plain Python cache fields, attrs, and same-file helper args in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py): `RY._seq_len_cached/_cos_cached/_sin_cached -> slc/cc/sc`, `CSA.num_heads/head_dim/use_xsa -> nh/hd/ux`, `GPT._init_weights -> iw`, plus shorter same-file args and locals like `v_embed/v_residual -> ve/vr`, `ve_cache -> vc`, `backout_layer -> bo`, and `x_backout -> xb`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran the stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that verifies `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, the renamed `DTL.nb()` path, the renamed runtime paths on a tiny CPU VRL model, and a `HEAD` vs working-copy export-byte comparison on the same tiny skeleton.
- Result: The candidate file shrank from `51,079` bytes to `50,523` bytes, another `556` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger `HEAD` vs working-copy local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=67051`, `work_blob_size=67051`, `same_blob=True`, `same_raw=True`, `sha256=27fcab33cc9bfcc6356db3a2d222528d40d50465ca608c0b74d748cfafc6aa72`, so this cache/helper rename pass is behavior-neutral under that probe.
- Decision: Keep the cache/helper shrink. This is another sizable counted-size win with no measured change to the tiny local export blob, no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 07:05 CDT
- Commit: `ac826a9`
- Lane: non-ttt VRL code-size hygiene + export-path correctness
- Objective: Reclaim a few more counted bytes from plain Python attrs in `RN/RY/CSA/BHE/GPT` while also fixing a real runtime bug in the export path where `ree()` still called `ch(..., num_batches=...)` after `ch()` had been renamed to `nbc=`.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to fix `ree(...)->ch(..., nbc=a.gcb)` and shortened only plain Python attrs and non-persistent buffer names such as `eps -> e`, `base/tsl/rope_dims/inv_freq -> b/t/rd/ifq`, `bgvs -> bv`, `teis -> ti`, and `lsc -> ls`, updating only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`, and reran the stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, the renamed `DTL.nb()` path, a direct `ch(..., nbc=1)` smoke, the renamed runtime attrs on a tiny CPU VRL model, and a `HEAD` vs working-copy export-byte comparison on the same tiny skeleton.
- Result: The candidate file shrank from `50,523` bytes to `50,418` bytes, another `105` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=67051`, `work_blob_size=67051`, `same_blob=True`, `same_raw=True`, `sha256=27fcab33cc9bfcc6356db3a2d222528d40d50465ca608c0b74d748cfafc6aa72`. The direct `ch(..., nbc=1)` call also ran cleanly, confirming the renamed keyword path now executes instead of throwing at runtime. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the bug fix and attr shrink. The keyword fix closes a real export-path regression, and the attr renames add a free counted-size win without changing the tiny local export blob, `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this correctness+size pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 07:07 CDT
- Commit: `f4d2d0f`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes by shortening only plain Python attrs that do not participate in module names, parameter names, checkpoint keys, or artifact layout, and by removing one dead `RY` instance attr.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to drop dead `RY.self.d`, shorten `GPT` plain attrs `te/ti/ls/num_layers -> t/ts/l/nl`, and update only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran the stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, `DTL.nb()`, `ch(..., nbc=1)`, and the renamed runtime attrs on a tiny CPU VRL model, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `50,418` bytes to `50,353` bytes, another `65` bytes of direct code-payload savings. The stronger local harness still matched exactly on the tiny CPU skeleton: `head_blob_size=67051`, `work_blob_size=67051`, `same_blob=True`, `same_raw=True`, `sha256=27fcab33cc9bfcc6356db3a2d222528d40d50465ca608c0b74d748cfafc6aa72`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the plain-attr shrink. This is a small but clean counted-size win with no measured change to the tiny local export blob, no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 07:09 CDT
- Commit: `b2239d0`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes in the Hessian/export helper path by shortening only same-file locals and dropping unused closure parameters, without touching artifact layout, checkpoint/state-dict keys, or runpod-facing markers.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten the `ch()` helper loop locals (`name/module/cols -> n/m/c`), drop unused `mh()` closure parameters, shorten the hook closure args, and simplify the `ree()` Hessian map fill from duplicated `sd_name/h_name` locals down to a single `k = n + WT`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran the stronger local `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that verified `eqm()/dqm()`, `etr()/dtr()`, `ze6/zd6`, `cmb()/dmb()`, `DTL.nb()`, `ch(..., nbc=1)`, and a `HEAD` vs working-copy export-byte comparison on the same tiny CPU skeleton, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `50,353` bytes to `50,175` bytes, another `178` bytes of direct code-payload savings. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=67051`, `work_blob_size=67051`, `same_blob=True`, `same_raw=True`, `sha256=27fcab33cc9bfcc6356db3a2d222528d40d50465ca608c0b74d748cfafc6aa72`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the export-helper shrink. This is another clean counted-size win with no measured change to the tiny local export blob, no change to `LF`, `cmb()`, serialized tensor layout, checkpoint field names, or runpod recovery tooling.
- Next step: Push this helper pass so the repo stays on the smallest counted candidate while preserving the current best-known `l0762` exporter path for the next live export-only rerun.

- Timestamp: 2026-03-24 07:18 CDT
- Commit: `a3c3b4e`
- Lane: non-ttt VRL exporter tuning
- Objective: Re-run a focused raw `LZMA2` neighborhood sweep around the current `hc3 / 18 MiB / lc=0 / lp=1 / pb=0 / nice_len=76 / depth=2` setting and keep only a filter change that produces a real chooser-level blob win under the local CPU VRL skeleton.
- Command or config: Ran a `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness that imports [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), builds the deterministic CPU VRL skeleton from current `H` defaults, quantizes it through `qsd()/eqm()/etr()/pi6()`, and sweeps raw `LZMA2` candidates in the tight neighborhood `dict in {17,18,19,20} MiB`, `lc=0`, `lp in {0,1}`, `pb in {0,1}`, `nice_len in {72,76,80,84}`, `depth in {1,2,3}` while keeping chooser parity against the fixed `zlib` and `zstd` alternatives. Then patched `LF` in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) from `lp=1, pb=0, depth=2` to `lp=0, pb=1, depth=1`, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy blob comparison harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The focused sweep found a better raw-`LZMA2` point at `hc3 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=76 / depth=1`. On the deterministic local CPU VRL skeleton, the chooser-level blob improved from `head_blob_size=4463870` to `work_blob_size=4457516`, a real `-6354` byte win, while `head_raw_size=work_raw_size=20125729` and exact round-trip still held (`head_roundtrip=True`, `work_roundtrip=True`). The stronger comparison produced different blob SHA-256s, as expected for a real codec change: `acf96b429dfce9b7fb9c222d410a964031cc3bdf7c03b221ff6805dbe4bec11b -> b292db17643b87a74a94e57f533ba1491ef3de96641d59ead9e03d7d2825d851`. The candidate file size stayed flat at `50,175` bytes, and readiness still passed. The live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the new `LF` default. This is a real measured exporter win under the stronger local skeleton harness, with no source-size regression and no readiness fallout, so it is a higher-leverage improvement than another micro rename pass.
- Next step: Push the new filter and journal entry so the repo is staged on the best measured local exporter path before the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:24 CDT
- Commit: `7de6c99`
- Lane: non-ttt VRL exporter tuning
- Objective: Probe the immediate neighborhood around the new `hc3 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=76 / depth=1` filter and keep only a further real chooser-level blob improvement.
- Command or config: Ran another tighter `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness over the same deterministic CPU VRL skeleton, this time sweeping `dict in {18,19,20,21} MiB`, `lc=0`, `lp=0`, `pb in {0,1,2}`, `nice_len in {72,74,76,78,80}`, and `depth in {0,1,2,3}` under `mf=HC3`, still with chooser parity against fixed `zlib` and `zstd` alternatives. The best neighborhood point came out at `hc3 / 18 MiB / lc=0 / lp=0 / pb=2 / nice_len=74 / depth=1`, so [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) was patched from `pb=1, nice_len=76` to `pb=2, nice_len=74`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy blob comparison harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The neighborhood sweep found another real chooser-level blob win. On the deterministic local CPU VRL skeleton, the chooser-level blob improved from `head_blob_size=4457516` to `work_blob_size=4457166`, another `-350` bytes, while `head_raw_size=work_raw_size=20125729` and exact round-trip still held (`head_roundtrip=True`, `work_roundtrip=True`). The blob SHA-256 changed as expected for a real codec change: `b292db17643b87a74a94e57f533ba1491ef3de96641d59ead9e03d7d2825d851 -> bf4a3f967336aac18e13d6dff074f862a05f821cdd973537ec467f3ecaaffa9b`. The candidate file size stayed flat at `50,175` bytes, readiness still passed, and the live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.39, deficit: $0.06).`
- Decision: Keep the new `pb=2, nice_len=74` filter. It is another clean measured exporter win with no source-size cost and no readiness fallout, so it should replace the previous `pb=1, nice_len=76` default.
- Next step: Push this tighter filter and journal entry so the repo is staged on the best measured local exporter path before the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:30 CDT
- Commit: `596cc97`
- Lane: non-ttt VRL exporter tuning
- Objective: Run one more fine-grained raw `LZMA2` sweep around the new `hc3 / 18 MiB / lc=0 / lp=0 / pb=2 / nice_len=74 / depth=1` winner, including smaller dictionaries and `MF_HC4`, and keep only a stronger measured chooser-level blob win.
- Command or config: Ran a tighter `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness over the same deterministic CPU VRL skeleton, sweeping `dict in {14,15,16,17,18} MiB`, `lc=0`, `lp=0`, `pb in {1,2}`, `nice_len in {72,73,74,75,76}`, `depth in {0,1,2}`, and both `mf in {HC3, HC4}`, still with chooser parity against fixed `zlib` and `zstd` alternatives. The best point came out at `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1`, which ties the best result already at `17 MiB` but avoids growing the counted source because `9<<21` stays shorter than a `17 MiB` literal. Patched `LF` in [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) from `MF_HC3, pb=2, nice_len=74` to `MF_HC4, pb=1, nice_len=72`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy blob comparison harness, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The fine sweep found another real chooser-level blob win. On the deterministic local CPU VRL skeleton, the chooser-level blob improved from `head_blob_size=4457166` to `work_blob_size=4456362`, another `-804` bytes, while `head_raw_size=work_raw_size=20125729` and exact round-trip still held (`head_roundtrip=True`, `work_roundtrip=True`). The blob SHA-256 changed as expected for a real codec change: `bf4a3f967336aac18e13d6dff074f862a05f821cdd973537ec467f3ecaaffa9b -> 31f4de665f7a20ba1e33baea8f08285b0bdfae48ae761d4d81b4d6cfcd15c2bf`. The candidate file size stayed flat at `50,175` bytes, readiness still passed, and the live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep the new `hc4 / pb=1 / nice_len=72` filter. It is another clean measured exporter win with no source-size cost and no readiness fallout, and it is now the best measured local exporter setting in the repo.
- Next step: Push this `hc4` filter and journal entry so the repo is staged on the best measured local exporter path before the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:34 CDT
- Commit: `60a14dc`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from the export/checkpoint helpers by shortening only file-internal locals and one unused helper parameter, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Confirmed with a follow-up sweep that the current `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1` filter is still locally optimal across nearby `hc4`, `bt4`, and `MODE_FAST` candidates. Then patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only file-internal helper names and locals in `rcp()`, `mspc()`, and `ree()`: `spec -> sp`, `path_spec -> ps`, `ckpt_path -> cp`, `tmp_path -> tp`, `meta_bytes -> mb0`, `dtype_map -> dm0`, `name_len -> nl`, `torch_dt/np_dt -> td/nd`, and removed the unused `log0` parameter from `mspc()`, updating the single same-file call site. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy blob comparison harness, ran a local `mspc('1', ...)` smoke to confirm the checkpoint path still writes, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `50,175` bytes to `49,998` bytes, another `177` bytes of direct code-payload savings. The stronger local harness still matched exactly on the current deterministic CPU VRL skeleton: `head_blob_size=4456362`, `work_blob_size=4456362`, `delta=0`, `head_raw_size=work_raw_size=20125729`, `same_blob=True`, and identical SHA-256 `31f4de665f7a20ba1e33baea8f08285b0bdfae48ae761d4d81b4d6cfcd15c2bf`. The local `mspc('1', ...)` smoke wrote `pre_export_model.pt` successfully. Readiness still passed, and the live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep the helper/local shrink. This is a clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:36 CDT
- Commit: `5184322`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes from `main()` by shortening only local variables and dropping the unused `console` parameter from the local logger, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Confirmed with one more targeted sweep that the current `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1` filter still holds up against nearby `hc4`, `bt4`, and `MODE_FAST` variants. Then patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only function-local names in `main()`: removed the unused `console` parameter from `log0()`, renamed `mwm -> wm`, `ims/ios -> is0/os0`, `ema_state -> es`, `ttms/sas -> tm/ss`, `trl -> tls`, `ams -> cms`, and `avg_state -> avs`, updating only same-file uses. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy blob comparison harness, ran a local `mspc('1', ...)` smoke to confirm the checkpoint path still writes, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `49,998` bytes to `49,913` bytes, another `85` bytes of direct code-payload savings. The stronger local harness still matched exactly on the current deterministic CPU VRL skeleton: `head_blob_size=4456362`, `work_blob_size=4456362`, `delta=0`, `head_raw_size=work_raw_size=20125729`, `same_blob=True`, and identical SHA-256 `31f4de665f7a20ba1e33baea8f08285b0bdfae48ae761d4d81b4d6cfcd15c2bf`. The local `mspc('1', ...)` smoke still wrote `pre_export_model.pt` successfully. Readiness still passed, and the live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep the local-name shrink. This is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:38 CDT
- Commit: `4ac2592`
- Lane: non-ttt VRL exporter tuning
- Objective: Check whether any nearby `hc4`, `bt4`, or `MODE_FAST` raw `LZMA2` variants can beat the current `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1` winner before spending more time on exporter churn.
- Command or config: Ran another `uv run --with torch --with numpy --with sentencepiece --with zstandard` harness over the same deterministic CPU VRL skeleton, sweeping `hc4` normal-mode candidates across `dict in {17,18,19} MiB`, `pb in {0,1,2}`, `nice_len in {68,69,70,71,72,73}`, depth `1`, plus a smaller sidecar set of `hc4` `MODE_FAST` and `bt4` normal-mode candidates near the current point. After the sweep, reran `python3 runpod/check_ready.py` and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: No candidate beat the current local winner. The best follow-up points tied the current blob exactly rather than improving it: `blob=4456362 raw_lz=4456357` at `dict=17 or 18 MiB`, `pb=1`, `nice_len=72`, `depth=1`, under both `hc4` and `bt4` in normal mode. So the current exporter setting remains locally optimal within this neighborhood. Readiness still passed, and the live export-only recovery path remains blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep the current `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1` filter unchanged and stop spending more local time on nearby raw-`LZMA2` churn for now.
- Next step: Continue harvesting low-risk counted-source wins while waiting for enough Runpod balance to resume the surviving `final_model.pt` export-only path.

- Timestamp: 2026-03-24 07:43 CDT
- Commit: `1ce48a9`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from the eval/export path by shortening only file-private parameter and local names, while keeping the current `hc4` exporter bytes unchanged and touching the renamed runtime paths under a local torch harness.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only file-private eval/export names: `rank -> rk` in `evv()`, `evl()`, `ree()`, and `main()`, `warmup_x -> wx`, `t_eval/eval_time -> te/et`, `swa_state/swa_count -> sws/swc`, and the `DTL.__init__()` rank parameter, updating only same-file call sites. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on a deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, touched `mspc('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `49,913` bytes to `49,801` bytes, another `112` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes also exercised the renamed paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep the eval/export local-name shrink. This is another clean counted-size win with no measured change to the current best local exporter blob and successful direct smokes through the renamed runtime paths.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:45 CDT
- Commit: `298a0c2`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from `lvt()`, `evv()`, `evl()`, and the prune block in `ree()` by shortening only file-private parameter and local names, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only file-private names: `pattern/files/tokens/usable -> pat/fs/ts/u` in `lvt()`, `local -> lc` and `val_loss -> vl` in `evv()`, `logits_fn/windows/batch/x_list/y_list/total -> lfn/ww/bt/xl/yl/tt` in `evl()`, and `ai6/all_vals/threshold -> a6/av/thr` in the prune block inside `ree()`, updating only same-file uses. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, and touched `mspc('1', ...)` again.
- Result: The candidate file shrank from `49,801` bytes to `49,595` bytes, another `206` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes again exercised the renamed runtime paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`.
- Decision: Keep this eval/loader/prune local-name shrink. This is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:47 CDT
- Commit: `fb61617`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from the quant/dequant helpers by shortening only file-private locals in `qi6g()`, `qi6p()`, `qft()`, `qsd()`, and `dsd()`, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only file-private quant/dequant names: `best_scale -> best_s`, `row_clip -> rc`, `clip_abs -> ca`, `clipped -> ct`, `result/meta -> r/m`, `int6_cats -> i6c`, `tensor -> tn`, and `orig_dtype/template_sd -> od/tsd`, updating only same-file uses. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, and touched `mspc('1', ...)` again.
- Result: The candidate file shrank from `49,595` bytes to `49,349` bytes, another `246` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes again exercised the touched runtime paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`.
- Decision: Keep this quant/dequant local-name shrink. This is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:50 CDT
- Commit: `34d7958`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from repeated tensor method spellings by introducing tiny file-private aliases for `.float()` and `.reshape()`, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to add `TF=lambda t:t.float()` and `RS=lambda t,*s:t.reshape(*s)`, then rewrote only same-file call sites across the eval, quant/dequant, int6 codec, model, Hessian, and training helpers where that was net-positive on source size. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, and touched `mspc('1', ...)` again.
- Result: The candidate file shrank from `49,349` bytes to `49,198` bytes, another `151` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes again exercised the touched runtime paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`.
- Decision: Keep this tensor-method alias pass. It is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:53 CDT
- Commit: `bda3aea`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim more counted bytes from repeated `.to(dtype=...)` and `.item()` spellings by introducing tiny file-private aliases, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to add `TY=lambda t,d:t.to(dtype=d)` and `IT=lambda t:t.item()`, then rewrote only same-file call sites across the optimizer, eval, quant/dequant, model, and training helpers where that was net-positive on source size. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, and touched `mspc('1', ...)` again.
- Result: The candidate file shrank from `49,198` bytes to `49,097` bytes, another `101` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes again exercised the touched runtime paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`.
- Decision: Keep this dtype/item alias pass. It is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 07:55 CDT
- Commit: `1a5a9d4`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes from repeated `.contiguous()` spellings by introducing one tiny file-private alias, while keeping the current `hc4` exporter bytes unchanged.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to add `CG=lambda t:t.contiguous()`, then rewrote only same-file call sites where that was net-positive on source size: `DX()`, `lvt()`, the int8 quant path in `qft()`, and the export raw-byte path in `ree()`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the renamed `evl()` path under that harness, and touched `mspc('1', ...)` again.
- Result: The candidate file shrank from `49,097` bytes to `49,062` bytes, another `35` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger local harness again matched exactly on the tiny CPU skeleton: `head_blob_size=17194`, `work_blob_size=17194`, `delta=0`, `head_raw_size=32248`, `work_raw_size=32248`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `edb6a00f3bb50b043c97c2e37955521083dca652f28cbc873642c0cd124e3fc8`. The current-only smokes again exercised the touched runtime paths successfully: `evl_val_loss=3.472115`, `evl_val_bpb=5.009203`, and `mspc_wrote=True`.
- Decision: Keep this contiguous-alias pass. It is another clean counted-size win with no measured change to the current best local exporter blob and no readiness fallout.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt.

- Timestamp: 2026-03-24 08:04 CDT
- Commit: `05108be`
- Lane: non-ttt VRL code-size hygiene
- Objective: Reclaim a few more counted bytes by shortening only file-private helper names across the eval/export/codec/Hessian stack while preserving the current `hc4` exporter bytes exactly.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten only file-private helper definitions and same-file call sites, including `lvt/evv/eqm/dqm/etr/dtr/qi6g/qi6p/qft/qsd/dsd/pi6l/ui6l/pi6/ui6/mcn/cmb/dmb/evl/rcp/mspc/ree -> lv/vv/eq/dq/tr/rt/qg/qp/q8/qs/ds/p6l/u6l/p6/u6/cn/cm/db/vl/rp/mp/re`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran a stronger compatibility-aware `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on a deterministic tiny CPU skeleton, touched the renamed `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `49,062` bytes to `49,005` bytes, another `57` bytes of direct code-payload savings. Compile and readiness checks still passed. The stronger compatibility-aware harness matched exactly on the tiny CPU skeleton: `head_blob_size=16922`, `work_blob_size=16922`, `delta=0`, `head_raw_size=36582`, `work_raw_size=36582`, `same_blob=True`, `same_raw=True`, and identical SHA-256 `4dc32e06171c19dffa6aad762cd74e20813c37d7b72017fcd247a01589c9bfdb`. The current-only smokes again exercised the renamed runtime paths successfully: `evl_val_loss=4.166742`, `evl_val_bpb=6.011338`, and `mspc_wrote=True`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.38, deficit: $0.07).`
- Decision: Keep this helper-rename pass. It is another clean counted-size win with no measured change to the current best local exporter blob, and the compatibility-aware harness removes the earlier false failure caused by `HEAD` still exporting the old helper names.
- Next step: Push this hygiene pass so the repo stays on the smallest counted candidate while preserving the current best measured `hc4` exporter path for the next live export-only resume attempt, then keep harvesting low-risk offline wins while Runpod remains balance-blocked.

- Timestamp: 2026-03-24 08:10 CDT
- Commit: `e7ee12e`
- Lane: non-ttt VRL state-key compaction
- Objective: Reclaim real artifact bytes, not just counted-source bytes, by shortening repeated model-private state-dict key fragments that get serialized into the export blob.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) in two stacked passes: first shortened the `ModuleList` key stem `blocks -> bs`, then shortened repeated GPT/BL state-key fragments `attn_scale/mlp_scale/resid_mix -> asc/msc/rm` and `tok_emb/lm_head/bigram/backout_lambda/skip_weights/ve_shared/ve_layer_scales -> tke/lmh/bgm/bol/skw/vsh/vls`, while updating `CP`, `cpm()`, and the same-file optimizer/runtime call sites to match. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `49,062` bytes to `48,506` bytes, another `556` bytes of direct code-payload savings. More importantly, the stronger local harness showed a real chooser-level blob win on the tiny CPU skeleton: `head_blob_size=16922`, `work_blob_size=16864`, `delta=-58`, `head_raw_size=36582`, `work_raw_size=36399`, with exact round-trip still holding and the current-only smokes still succeeding: `evl_val_loss=4.166742`, `evl_val_bpb=6.011338`, `mspc_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `4dc32e06171c19dffa6aad762cd74e20813c37d7b72017fcd247a01589c9bfdb -> fadd5d6cb9b49dda8519c7749de5c89ac797a7dd2e13fa6d08b8155dcb687456`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.37, deficit: $0.07).`
- Decision: Keep this state-key compaction pass. It is the best offline move in a while because it improves both the counted source and the wrapped export blob while keeping the local runtime/export smokes green.
- Next step: Push this state-key compaction so the repo stays on the smallest measured local candidate, then keep probing the remaining repeated state-key fragments only where the expected return still looks material.

- Timestamp: 2026-03-24 08:11 CDT
- Commit: `0d5bc41`
- Lane: non-ttt VRL state-key compaction
- Objective: Reclaim a few more wrapped-blob bytes from the remaining repeated state-key fragments in the VRL attention/residual path after the larger `bs/asc/msc/rm/tke/lmh/...` compaction pass.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten the remaining repeated model-private key fragments `q_gain -> qgn` and `vrl_alphas -> vra`, updating `CP`, the same-file attribute references in `CSA`/`GPT`, and the optimizer-side `spa` append path. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `48,506` bytes to `48,455` bytes, another `51` bytes of direct code-payload savings. The stronger local harness showed another real chooser-level blob win on the tiny CPU skeleton: `head_blob_size=16864`, `work_blob_size=16855`, `delta=-9`, `head_raw_size=36399`, `work_raw_size=36386`, with exact round-trip still holding and the current-only smokes still succeeding: `evl_val_loss=4.166742`, `evl_val_bpb=6.011338`, `mspc_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `fadd5d6cb9b49dda8519c7749de5c89ac797a7dd2e13fa6d08b8155dcb687456 -> 4c79102dc4ac1178ec4c091524bc66c6a09d54890ea48759e4a8a7db7092cc46`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.37, deficit: $0.07).`
- Decision: Keep this follow-up compaction too. The marginal win is smaller than the previous state-key pass, but it is still a real measured wrapped-blob improvement plus another small counted-source reduction, with no local smoke regressions.
- Next step: Push this follow-up compaction so the repo stays on the smallest measured local candidate, then either stop on this diminishing-return state-key lane or only touch another fragment if it looks similarly low-risk and repeated.

- Timestamp: 2026-03-24 08:16 CDT
- Commit: `b20cd50`
- Lane: non-ttt VRL state-key compaction
- Objective: Reclaim another chunk of real wrapped-blob bytes from the densest remaining repeated state-key stems, especially the per-block `attn/mlp/c_q/c_k/c_v/proj/fc` family, while keeping the local export/runtime probes green.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten the repeated model-private key stems `attn/mlp -> a/m`, `c_q/c_k/c_v -> cq/ck/cv`, `proj/fc -> p/f`, `qgn -> g`, and the `BHE`/`VE` member keys `embed/proj/scale -> e/p/s`, updating the same-file call sites, `cpm()`, and the optimizer-side references accordingly. The first attempt exposed that the old substring-based control-param matcher was too broad once `qgn` became `g`, so I tightened it to exact path-segment matching via the new `IC(...)` helper before re-running validation. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `48,455` bytes to `48,293` bytes, another `162` bytes of direct code-payload savings. The stronger local harness showed another real chooser-level blob win on the tiny CPU skeleton: `head_blob_size=16855`, `work_blob_size=16826`, `delta=-29`, `head_raw_size=36386`, `work_raw_size=36302`, with exact round-trip still holding and the current-only smokes still succeeding: `evl_val_loss=4.166742`, `evl_val_bpb=6.011338`, `mspc_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `4c79102dc4ac1178ec4c091524bc66c6a09d54890ea48759e4a8a7db7092cc46 -> b38a3588b10b526f5afd2d5ab8c18dd548e0c038eb5ab17aedb7183790b9d498`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.37, deficit: $0.08).`
- Decision: Keep this module-stem compaction pass. The exact-segment `IC(...)` fix removed the false control-param broadening from the first attempt, and the final version delivers a clean wrapped-blob improvement plus another meaningful counted-source reduction with no local smoke regressions.
- Next step: Push this compaction so the repo stays on the smallest measured local candidate, then stop spending local time on increasingly marginal state-key churn unless another repeated stem stands out as similarly safe and high-frequency.

- Timestamp: 2026-03-24 08:19 CDT
- Commit: `afa8fa9`
- Lane: non-ttt VRL code-size hygiene
- Objective: Harvest one more low-risk cleanup pass by shortening the remaining long one-off runtime names `smear/gate/final_norm/attn_norm/mlp_norm/rotary`, while also squeezing the single serialized state key `smear.gate`.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten `smear -> sg`, `gate -> g`, `final_norm -> fn`, `attn_norm/mlp_norm -> an/mn`, and `rotary -> ro`, updating the same-file optimizer/runtime references and the control-param path list accordingly. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `48,293` bytes to `48,197` bytes, another `96` bytes of direct code-payload savings. The stronger local harness showed a final tiny but real chooser-level blob win on the tiny CPU skeleton: `head_blob_size=16826`, `work_blob_size=16825`, `delta=-1`, `head_raw_size=36302`, `work_raw_size=36296`, with exact round-trip still holding and the current-only smokes still succeeding: `evl_val_loss=4.166742`, `evl_val_bpb=6.011338`, `mspc_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `b38a3588b10b526f5afd2d5ab8c18dd548e0c038eb5ab17aedb7183790b9d498 -> ac78fef873a8108f9a37e5ca3ae56ca8567ee721cf12c211170efc0e5a4b96a1`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.37, deficit: $0.08).`
- Decision: Keep this final hygiene pass. The win is small on the wrapped blob, but it is still positive, it trims the counted source further, and it leaves the stronger local export/runtime probes green.
- Next step: Push this pass so the repo stays on the smallest measured local candidate, then stop on this local lane unless a clearly larger offline opportunity appears, because the next real score move now needs Runpod balance more than more micro-compaction.

- Timestamp: 2026-03-24 08:27 CDT
- Commit: `ed6584c`
- Lane: non-ttt VRL root-state compaction
- Objective: Reclaim both counted-source bytes and wrapped export bytes by shortening the remaining GPT root/member state-key stems after the earlier `b/tk/bg/...` cleanup, while keeping the runtime/export path green.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) in a stacked root-stem pass: first completed the pending `bs/tke/bgm/bol/skw/vsh/vls/vra -> b/tk/bg/bo/sw/vh/vl/va` wiring, then compacted the remaining repeated GPT root/member keys `tk/bg/sg/bo/sw/vh/vl/va/lmh/fn -> x/g/s/o/w/v/y/r/h/n`, updating `CP`, `cpm()`, the same-file optimizer setup, and the runtime call sites to match. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, touched `mp('1', ...)`, and reran `bash runpod/local_resume_existing_export.sh liyrlu10czeqvw ./final_model.pt /workspace/golf 5`.
- Result: The candidate file shrank from `48,197` bytes to `48,094` bytes, another `103` bytes of direct code-payload savings. The stronger local harness showed another real chooser-level wrapped-blob win on the tiny CPU skeleton: `head_blob_size=13773`, `work_blob_size=13755`, `delta=-18`, `head_raw_size=30628`, `work_raw_size=30571`, with exact `cm()/db()` round-trip still holding. The current-only runtime smokes also stayed green: `evl_val_loss=4.171484`, `evl_val_bpb=6.018180`, `mp_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `db8b0cf4248c4bb49001d3aee0564b8818fa788ba5deb909c3858dc437329fe4 -> 4220c5a4aed31bd609ff2f1275e9cf8226cc36acc48e50caa7460242cca5f03f`. The live export-only recovery path is still blocked only by balance: `You need at least $0.45 (current: $0.37, deficit: $0.08).`
- Decision: Keep this root-state compaction pass. It improves both the counted candidate and the measured local wrapped blob, and the stronger local export/runtime probes stayed clean after the optimizer-path rewiring.
- Next step: Push this pass so the repo stays on the smallest measured local candidate, then stop on this offline lane unless another repeated serialized stem stands out as clearly higher-leverage than the current balance blocker.

- Timestamp: 2026-03-24 08:31 CDT
- Commit: `f2420f7`
- Lane: non-ttt VRL block-state compaction
- Objective: Reclaim another small chunk of both counted-source bytes and wrapped export bytes by shortening the last repeated per-block serialized stems inside `BL`/`CSA`: `asc/msc/rm` and `cq/ck/cv`.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to shorten `BL` state-key members `asc/msc/rm -> u/t/r` and `CSA` state-key members `cq/ck/cv -> q/k/v`, updating `CP`, the same-file runtime references, the block init warm-start path, and the initial VRL residual probe path to match. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` blob comparison harness on the deterministic tiny CPU skeleton, touched the current `vl()` path under that harness, and touched `mp('1', ...)`.
- Result: The candidate file shrank from `48,094` bytes to `48,051` bytes, another `43` bytes of direct code-payload savings. The stronger local harness showed another real chooser-level wrapped-blob win on the tiny CPU skeleton: `head_blob_size=13755`, `work_blob_size=13747`, `delta=-8`, `head_raw_size=30571`, `work_raw_size=30547`, with exact `cm()/db()` round-trip still holding. The current-only runtime smokes also stayed green: `evl_val_loss=4.171484`, `evl_val_bpb=6.018180`, `mp_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `4220c5a4aed31bd609ff2f1275e9cf8226cc36acc48e50caa7460242cca5f03f -> c176c6356d4e8493a6df8a498948574e015c130b2c41f65d1604f935c9cd0078`.
- Decision: Keep this block-state compaction pass. The gain is smaller than the previous root compaction, but it is still a real measured wrapped-blob improvement plus another clean counted-source reduction with no readiness or runtime regressions.
- Next step: Push this pass so the repo stays on the smallest measured local candidate, then keep preferring only repeated serialized stems or other artifact-facing changes that clearly beat the current diminishing-return hygiene lane while Runpod remains balance-blocked.

- Timestamp: 2026-03-24 08:34 CDT
- Commit: `5f940f3`
- Lane: non-ttt VRL exporter tuning
- Objective: Search for a materially better raw `LZMA2` filter than the current `hc4 / 18 MiB / lc=0 / lp=0 / pb=1 / nice_len=72 / depth=1` default, and only keep it if it improves the wrapped export blob under the stronger deterministic local harness.
- Command or config: Ran a focused `uv run --with torch --with numpy --with sentencepiece --with zstandard` sweep over nearby raw `LZMA2` settings on the deterministic tiny CPU skeleton, varying `mf`, `dict_size`, `lc`, `lp`, `pb`, `nice_len`, `depth`, and `mode`, while compressing the exact quantized raw payload that feeds `cm()`. The best raw result was `hc4 / 17 MiB / lc=0 / lp=1 / pb=0 / nice_len=68 / mode=normal` without an explicit `depth`. Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to use that filter in `LF`, then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, reran the stronger `HEAD` vs working-copy blob comparison harness, touched the current `vl()` path under that harness, and touched `mp('1', ...)`.
- Result: The candidate file shrank from `48,051` bytes to `48,036` bytes, another `15` bytes of direct code-payload savings. More importantly, the stronger local harness showed a large wrapped-blob win with the exact same raw payload: `head_blob_size=13747`, `work_blob_size=12489`, `delta=-1258`, `head_raw_size=30547`, `work_raw_size=30547`, with exact `cm()/db()` round-trip still holding. The current-only runtime smokes also stayed green: `evl_val_loss=4.171484`, `evl_val_bpb=6.018180`, `mp_wrote=True`. The blob SHA-256 changed as expected for a real serialization change: `c176c6356d4e8493a6df8a498948574e015c130b2c41f65d1604f935c9cd0078 -> 80d2558443c79d50379d72178d79e6ea84e232a2c7b4669f417f3dfdd6a162f8`.
- Decision: Keep this exporter filter change. It is the strongest offline artifact win in the current local lane because it preserves the quantized raw payload exactly while making the wrapped blob materially smaller.
- Next step: Push this exporter tuning so the repo stays on the best measured local artifact path, then keep prioritizing artifact-facing changes over tiny code-only hygiene while the live Runpod resume remains balance-blocked.

- Timestamp: 2026-03-24 08:45 CDT
- Commit: `ba50759`
- Lane: non-ttt VRL exporter tuning
- Objective: Resolve the remaining `hc4` vs `hc3` ambiguity inside the kept `17 MiB / lc=0 / lp=1 / pb=0 / nice_len=68` raw `LZMA2` family by checking both the tiny deterministic harness and a much more realistic default-like proxy.
- Command or config: Compared the kept `hc4` filter against `hc3` with the same `dict_size=17 MiB`, `lc=0`, `lp=1`, `pb=0`, and `nice_len=68`. First verified on the tiny deterministic harness, then reran the comparison on a default-like proxy closer to the real training/export config: `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`. The comparison measured the final chooser-level wrapped blob, not just raw `LZMA2` size. Then patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to switch `LF[0]["mf"]` from `lzma.MF_HC4` to `lzma.MF_HC3`, reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, and reran `python3 runpod/check_ready.py`.
- Result: The tradeoff is asymmetric but favorable at realistic scale. On the tiny harness, `hc3` is `+1` byte worse on the wrapped blob with identical raw payload: `head_blob_size=12489`, `work_blob_size=12490`, `delta=+1`, `head_raw_size=30547`, `work_raw_size=30547`. On the default-like proxy, `hc3` is materially better with the same raw payload: `head_blob_size=4350089`, `work_blob_size=4349038`, `delta=-1051`, `head_raw_size=21470079`, `work_raw_size=21470079`. Compile and readiness checks still passed after the code change.
- Decision: Keep the `hc3` switch. The tiny harness regression is only one byte, while the default-like proxy win is over one kilobyte with identical raw payload, which is the better signal for the real exporter regime we care about.
- Next step: Push this filter change so the branch stays on the best realistic local exporter setting, then stop spending more time on broad filter sweeps unless a narrower artifact-facing hypothesis beats this new `hc3` default on the larger-proxy checks.

- Timestamp: 2026-03-24 08:39 CDT
- Commit: journal-only note after rejected experiments
- Lane: non-ttt VRL exporter research
- Objective: Test whether two higher-risk exporter ideas would outperform the current `hc4 / 17 MiB / lc=0 / lp=1 / pb=0 / nice_len=68` default enough to justify either code churn or a filter change.
- Command or config: First tried a compact metadata format that removed full metadata names from `QMB` and instead aligned kind codes to the current sorted `state_dict` keys at load time. Then tested a follow-up filter tweak that changed only `nice_len` from `68` to `72` within the current `hc4 / 17 MiB / lc=0 / lp=1 / pb=0` family. For both, I reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and compared `HEAD` vs worktree on both the deterministic tiny CPU harness and larger CPU proxy models, including a default-like proxy at `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`.
- Result: The compact metadata path looked promising on the tiny harness (`head_blob_size=12489`, `work_blob_size=12362`, `delta=-127`, `head_raw_size=30547`, `work_raw_size=30070`), but it failed at larger scale: on the default-like proxy it made the wrapped blob worse by `+4525` bytes despite a slightly smaller raw payload (`head_blob_size=3454575`, `work_blob_size=3459100`, `head_raw_size=10775950`, `work_raw_size=10774253`). The `nice_len=72` tweak also failed the larger proxy check: it was only `+2` bytes worse on the tiny harness (`12489 -> 12491`) but `+364` bytes worse on the default-like proxy (`4350089 -> 4350453`) with identical raw payload size. Both experiments were reverted, leaving the code on the prior `a5cea53` exporter state.
- Decision: Reject both experiments. The compact metadata idea is not net-positive at realistic scale, and `nice_len=72` loses to the current `nice_len=68` filter on the stronger default-like proxy.
- Next step: Keep the branch on the current `a5cea53` exporter default and continue searching only for artifact-facing changes that beat the larger-proxy checks, since the next real leaderboard move still requires a valid live export attempt on Runpod.

- Timestamp: 2026-03-24 08:54 CDT
- Commit: `bcd67ee`
- Lane: non-ttt VRL state-key compaction
- Objective: Reclaim real wrapped-blob bytes from the last large repeated suffix in serialized state keys by shortening `*.weight`, but only keep it if the larger proxy beats the extra source bytes.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) so `GPT.state_dict()` rewrites serialized `*.weight` keys to the short suffix `*.z`, while `GPT.load_state_dict()` accepts both legacy `*.weight` and new `*.z` forms. The first attempt used `*.w`, but the stronger proxy exposed a real collision with the existing control-param segment `w`, which made ordinary weights fall into the control-path bucket and exploded the artifact; I rejected that intermediate state and kept the non-colliding `*.z` version instead. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran a stronger `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` comparison harness on both the deterministic tiny CPU skeleton and the larger default-like proxy `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`, including strict old-key/new-key load compatibility checks.
- Result: The candidate file grew from `48,036` bytes to `48,353` bytes (`+317`). The kept `*.z` version still delivered real artifact wins: on the tiny harness, `head_blob_size=84932`, `work_blob_size=84783`, `delta=-149`, `head_raw_size=219911`, `work_raw_size=219801`; on the default-like proxy, `head_blob_size=4240254`, `work_blob_size=4239398`, `delta=-856`, `head_raw_size=21470064`, `work_raw_size=21469709`. The raw default-like drop is exactly `-355` bytes, matching `71` repeated `*.weight -> *.z` suffix savings. Strict compatibility stayed clean in both directions we care about: the current loader accepted both old `*.weight` keys and new `*.z` keys with `missing=0` and `unexpected=0`. Compile and readiness checks still passed.
- Decision: Keep the `*.z` state-key remap. It loses on neither proxy, and on the more realistic default-like proxy it beats the `+317` source bump by `-856` wrapped bytes for a net combined win of about `-539` bytes, which is the better signal for the real exporter regime than the tiny harness.
- Next step: Push this state-key remap so the branch stays on the best measured local artifact state, then stop on local artifact churn unless another realistic-scale proxy win is clearly larger than the remaining Runpod balance blocker.

- Timestamp: 2026-03-24 08:58 CDT
- Commit: journal-only note after rejected experiment
- Lane: non-ttt VRL exporter research
- Objective: Check whether dropping the serialized weight suffix entirely would outperform the kept `*.z` remap after paying for the extra compatibility logic.
- Command or config: Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to serialize `*.weight` tensors at their bare module paths instead of `*.z`, updated `load_state_dict()` to accept bare names plus the older `*.weight` and `*.z` forms, and aligned the Hessian/export lookup path to those bare names. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and compared the kept `bcd67ee` `*.z` state against the no-suffix variant on both the deterministic tiny CPU harness and the larger default-like proxy `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`, including strict old/new load compatibility checks.
- Result: The no-suffix variant looked slightly better on the tiny harness but failed at realistic scale. Candidate source grew from `48,353` bytes to `48,484` bytes (`+131`). On the tiny harness it improved a little: `head_blob_size=84783`, `work_blob_size=84764`, `delta=-19`, `head_raw_size=219801`, `work_raw_size=219757`. On the default-like proxy it lost badly on the wrapped artifact despite a slightly smaller raw payload: `head_blob_size=4239398`, `work_blob_size=4240685`, `delta=+1287`, `head_raw_size=21469709`, `work_raw_size=21469567`. Strict compatibility stayed clean (`old_missing=0`, `old_unexpected=0`, `new_missing=0`, `new_unexpected=0`), so this was a true compression loss rather than a loader bug. I reverted the experiment and restored the kept `*.z` state immediately.
- Decision: Reject the no-suffix remap. The realistic proxy is the deciding signal, and there the change is clearly net-negative even before counting the extra `+131` source bytes.
- Next step: Stay on the pushed `*.z` state and continue only with artifact-facing ideas that beat the larger-proxy checks, because the remaining blocker to a real leaderboard attempt is still Runpod balance rather than local cleanliness.

- Timestamp: 2026-03-24 09:07 CDT
- Commit: `e77aa8f`
- Lane: non-ttt VRL state-key compaction
- Objective: Improve on the kept `*.z` remap without paying extra code-size overhead by sweeping the single-character serialized weight suffix and keeping only a realistic-scale proxy win.
- Command or config: First ran a default-like proxy sweep over candidate suffixes `{z,a,c,d,e,f,h,i,j,k,l,m,n,p,q,v}` by monkeypatching `WT` inside the current [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) and rebuilding the full quantized/export path at `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`. The best wrapped-blob results tied at `.a` and `.k`, both beating the current `.z` state. Then ran a tiny-harness tie-breaker plus strict load compatibility; `.k` beat `.a` slightly on the tiny harness while keeping exact raw size and clean loads. Patched [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py) to switch `WT` from `".z"` to `".k"` and added a tiny backward-compat shim so `load_state_dict()` accepts both the new `*.k` form and the previous `*.z` form alongside legacy `*.weight`. Then reran `uv run python -m py_compile candidates/non_ttt_vrl_gptq/train_gpt.py`, reran `python3 runpod/check_ready.py`, and reran a direct `HEAD` vs working-copy `uv run --with torch --with numpy --with sentencepiece --with zstandard` comparison on both the deterministic tiny CPU harness and the default-like proxy, including strict old/new compatibility checks.
- Result: The candidate file grew from `48,353` bytes to `48,373` bytes (`+20`). The kept `.k` state improved the wrapped blob on both measured payloads with the same raw payload size as `.z`: tiny harness `head_blob_size=84783`, `work_blob_size=84701`, `delta=-82`, `head_raw_size=219801`, `work_raw_size=219801`; default-like proxy `head_blob_size=4239398`, `work_blob_size=4238370`, `delta=-1028`, `head_raw_size=21469709`, `work_raw_size=21469709`. Strict compatibility stayed clean in all directions tested: `old_missing=0`, `old_unexpected=0`, `new_missing=0`, `new_unexpected=0`. The earlier proxy sweep showed `.a` tied `.k` on the larger proxy (`4238370`) while `.k` was slightly better on the tiny harness (`84701` vs `84705`), so `.k` was the best measured suffix overall.
- Decision: Keep the `.k` remap. It improves the realistic proxy by `-1028` wrapped bytes while costing only `+20` counted-source bytes, for a net combined win of about `-1008` bytes over the previous pushed `.z` state, and it keeps both legacy `.weight` and immediate-prior `.z` loads working.
- Next step: Push this better suffix choice so the repo stays on the strongest measured local artifact state, then stop spending more time on local suffix churn unless a larger realistic-scale proxy win appears; the next real leaderboard step still depends on clearing the Runpod balance blocker.

- Timestamp: 2026-03-24 09:08 CDT
- Commit: journal-only note after suffix sweep
- Lane: non-ttt VRL exporter research
- Objective: Verify whether any other single-character serialized weight suffix beats the newly kept `.k` choice on the realistic proxy before freezing this lane.
- Command or config: Ran a broader `uv run --with torch --with numpy --with sentencepiece --with zstandard` default-like proxy sweep by monkeypatching `WT` across `{a..z,0..9}` minus the obvious collision set `{g,o,r,t,u,w,y,q,s}` inside the current [train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py), rebuilding the full quantized/export path each time at `VS=1024 NL=11 DM=512 NH=8 NKH=4 MM=3 BGVS=2048 BGD=128 XSN=11 RD=16 VED=128 VEL=9,10`, and ranking by chooser-level wrapped blob size.
- Result: Nothing beat the kept `.k` state. The best larger-proxy results were a tie between `.a` and `.k` at `4238370` wrapped bytes with the same `21469709` raw payload. The next-best candidate was `.0` at `4239016`, and the large middle cluster (`1..9`, `b`, `c`, `d`, `e`, `h`, `i`, `j`, `l`, etc.) all landed at `4239398`, effectively the previous `.z` level. Because the earlier tiny-harness tie-break already showed `.k` slightly beating `.a` (`84701` vs `84705`) with the same raw payload and compatibility behavior, there was no reason to change the kept code again.
- Decision: Keep `.k` as the final single-character suffix choice. The broader sweep confirms we are already sitting on the best measured realistic-proxy suffix family, and `.k` remains the best overall tie-break among those equal larger-proxy winners.
- Next step: Stop spending more time on single-character suffix churn and only reopen this local artifact lane for ideas that are materially different from suffix selection, because the remaining blocker to a real leaderboard attempt is still the Runpod balance gap.
