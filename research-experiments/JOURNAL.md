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

## 2026-03-22 21:43 CDT - Real Experiment Path Replaced The Fake Bootstrap Path

- The controller now distinguishes support jobs from real `job_kind=experiment` runs, so bootstrap snapshots no longer masquerade as a best score.
- Added a real runner at `scripts/run_record_experiment.py` that performs preflight checks, launches a candidate with `torch.distributed.run`, and writes `stats.json` for the controller.
- Added a data wrapper at `scripts/prepare_challenge_data.py` for downloading the official published FineWeb challenge shards into the cached official repo clone.
- The current lead candidate remains `record_candidates/2026-03-23_rank1_mixed_qat/`, which is derived from the official March 20 `#1` script and adds mixed STE fake-quant during training.
- Staged a real experiment manifest in `manifests/pending/rank1_mixed_qat_seed42_20260323.json`; the controller will ingest it only when runtime prerequisites are satisfied.
- A real preflight run was executed locally and wrote `runs/preflight_rank1_mixed_qat/stats.json`.
- The truthful blockers on this machine are: no CUDA devices, no `zstandard` module, and no downloaded challenge dataset/tokenizer yet.

## 2026-03-22 21:51 CDT - Colab VS Code Workflow Added

- Added `COLAB_VSCODE.md` and `colab_vscode_bootstrap.ipynb` so the repo has a notebook-first workflow that matches the official Google Colab VS Code extension.
- Made `research-agent/start_loop.sh`, `stop_loop.sh`, and `status_loop.sh` portable with `bash` shebangs so they can run inside a Colab Linux runtime.
- Extended `scripts/run_record_experiment.py` with `--required-cuda-devices`, `--set-env`, and `--set-env-file` so we can run smaller Colab pilot jobs without mutating the record candidate itself.
- Added `record_candidates/2026-03-23_rank1_mixed_qat/colab_pilot_env.json` with conservative single-GPU pilot overrides for Colab-backed validation runs.
- Confirmed the new Colab pilot preflight path works and reports truthful blockers for the current local machine rather than pretending the environment is ready.

## 2026-03-22 21:36 CDT - Rehydrated Rank-1 Script Into A Mixed-QAT Candidate

- Created `record_candidates/2026-03-23_rank1_mixed_qat/` by copying the current official `#1` March 20 script instead of reviving the older local baseline family.
- Added a mixed fake-quant path in `record_candidates/2026-03-23_rank1_mixed_qat/train_gpt.py` so exported block MLP matrices train against int5-style STE and exported attention / BigramHash projection matrices train against int6-style STE.
- Added `record_candidates/2026-03-23_rank1_mixed_qat/README.md` with the exact rank-1 provenance and the intended launch command.
- Ran `python3 -m py_compile record_candidates/2026-03-23_rank1_mixed_qat/train_gpt.py` successfully as a syntax check.
- Prepared the next run manifest as `manifests/pending/rank1_mixed_qat_seed42_20260323.json`, targeting `SEED=42` with `QAT_ENABLED=1`.
- Attempted to place that manifest in `../research-agent/loop/runtime/queue/pending/`, but the sandbox denied writes outside `research-experiments/` with `Operation not permitted`, so no controller-visible pending job was created this turn.
- Most likely next step: either widen writable access to the controller queue path or have the controller ingest the staged manifest from `research-experiments/manifests/pending/`, then run the mixed-QAT rank-1 variant and compare roundtrip `val_bpb` against `1.1428`.

## 2026-03-22 21:37 CDT - Git Metadata Writes Are Also Sandbox-Blocked

- Attempted `git add` and `git commit` for the new candidate files, but the sandbox could not create `/Users/aryan/Desktop/golf/.git/index.lock` and returned `Operation not permitted`.
- The worktree changes remain present under `research-experiments/`, but this Codex turn could not create a local checkpoint commit directly.
- Most likely next step: rely on the external controller's out-of-sandbox auto-commit path, or widen writable access to the repo `.git` directory so Codex can satisfy the frequent-commit rule itself.

## 2026-03-23T02:38:29.912352+00:00 - Controller Turn Summary: turn_20260323T023255Z

- Codex exit code: 0
- Decision: blocked
- Summary: Rehydrated the current official #1 script into a tracked candidate at `/Users/aryan/Desktop/golf/research-experiments/record_candidates/2026-03-23_rank1_mixed_qat/train_gpt.py` and added mixed STE fake-quant training for exported matrices: int5-style on MLP weights, int6-style on attention weights, and int6-style on the BigramHash projection. Added candidate docs, staged the intended seed-42 run manifest at `/Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_seed42_20260323.json`, syntax-checked the script with `python3 -m py_compile`, and appended the research log in `JOURNAL.md`.
- Best score bpb: None
- Best score source: None
- Job manifest created: True
- Job manifest path: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_seed42_20260323.json
- Repo changed: True
- Needs human attention: True
- Human message: Sandbox policy blocked writes to `/Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/pending/`, so I could not place the staged manifest into the live controller queue. It also blocked writes to `/Users/aryan/Desktop/golf/.git`, so manual `git add`/`git commit` failed with `.git/index.lock: Operation not permitted`. Please either widen writable access for those paths or make the external controller ingest manifests from `/Users/aryan/Desktop/golf/research-experiments/manifests/pending/`.
- Prompt: /Users/aryan/Desktop/golf/research-agent/loop/runtime/reports/turn_20260323T023255Z.prompt.md
- Log: /Users/aryan/Desktop/golf/research-agent/loop/runtime/logs/turn_20260323T023255Z.codex.log

## 2026-03-23T02:46:05.318883+00:00 - Manifest Waiting For Runtime: rank1_mixed_qat_seed42_20260323

- Path: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_seed42_20260323.json
- missing python module: torch
- missing python module: numpy
- missing python module: sentencepiece
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/datasets/fineweb10B_sp1024
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model
- missing python module: torch

## 2026-03-23T02:46:53.447367+00:00 - Manifest Waiting For Runtime: rank1_mixed_qat_seed42_20260323

- Path: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_seed42_20260323.json
- missing python module: torch
- missing python module: numpy
- missing python module: sentencepiece
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/datasets/fineweb10B_sp1024
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model

## 2026-03-23T02:52:23.620679+00:00 - Manifest Waiting For Runtime: rank1_mixed_qat_warmdown_ramp_seed42_20260323

- Path: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_warmdown_ramp_seed42_20260323.json
- missing python module: torch
- missing python module: numpy
- missing python module: sentencepiece
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/datasets/fineweb10B_sp1024
- missing path: /Users/aryan/Desktop/golf/research-experiments/cache/openai-parameter-golf/data/tokenizers/fineweb_1024_bpe.model

## 2026-03-22 21:58 CDT - Controller Observability Fix

- The older sandbox-blocked journal notes were historical from before the controller was updated to ingest staged manifests from `research-experiments/manifests/pending/`.
- Updated `research-agent/loopctl.py` so the supervisor records the active in-flight Codex turn, includes it in heartbeat data, and surfaces it in `status` output instead of only showing the last completed turn.
- Updated the loop launch scripts to use unbuffered Python so controller stdout reflects live progress more reliably during long autonomous turns.

## 2026-03-22 22:07 CDT - Colab Terminal Readiness

- Confirmed the Google Colab VS Code extension is installed locally as `google.colab-0.4.1`.
- Added `.vscode/settings.json` to pre-enable the experimental `colab.serverMounting` and `colab.terminal` features for this workspace.
- Added `research-experiments/scripts/colab_gpu_smoketest.sh` to verify that a shell is actually attached to the Colab runtime and to run a truthful 1-GPU preflight.
- Added `research-experiments/scripts/run_colab_pilot.sh` so once the Colab terminal is open we can launch the single-GPU pilot directly.
- Corrected the helper scripts to run from the `research-experiments/` root so their `stats.json` outputs land in `research-experiments/runs/` instead of a nested duplicate path.

## 2026-03-22 22:11 CDT - Prompt Reality Rules And External Repo Mirrors

- Tightened the research-agent goal and prompt so fake data, filler data, proxy metrics, fabricated scores, and submission-invalid shortcuts cannot be represented as real progress.
- Mirrored `https://github.com/thwu1/parameter-golf.git` into `research-experiments/cache/external/thwu1-parameter-golf/` at `45bbccff356439d2f0b0dbae06cc3fa58b9576ed` and marked it as the default record-track starting point.
- Mirrored `https://github.com/aruniyer/parameter-golf.git` into `research-experiments/cache/external/aruniyer-parameter-golf/` at `954a158102ec64c292ad82b2442e387e505a9388` as a third-place reference implementation.
- Added `research-experiments/scripts/sync_external_public_repos.sh` so fresh machines and Colab runtimes can recreate those mirrors even though `cache/` is gitignored.

## 2026-03-22 22:15 CDT - Fresh Clone Data Bootstrap Fix

- Fixed `research-experiments/scripts/prepare_challenge_data.py` so it automatically clones the official `openai/parameter-golf` repo into `research-experiments/cache/openai-parameter-golf/` when that cache is missing on a fresh machine such as Colab.
- This removes the old failure mode where data prep immediately crashed with `Missing helper script: .../cache/openai-parameter-golf/data/cached_challenge_fineweb.py`.

## 2026-03-22 22:44 CDT - Mac MLX Proxy Lane Using uv

- Added a Mac-native proxy workflow in `research-experiments/MAC_PROXY.md` plus Apple Silicon helper scripts under `research-experiments/scripts/` for `uv` environment setup, official data prep, smoke runs, frontier runs, and queued frontier manifests.
- Copied the public `thwu1` MLX trainer into `research-experiments/mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy/train_gpt_mlx.py` as the tracked local proxy starting point, with smoke/frontier env presets beside it.
- Created `.venv-mac` with `uv`, installed MLX and data dependencies, and downloaded the official `sp1024` tokenizer plus `8` local train shards for a stronger Mac proxy signal.
- Started `thwu1_mlx_mac_smoke_20260322_223016` on the official data. It is training cleanly on Apple Silicon with MLX at roughly `15k-16k tok/s`; the full validation pass is still running as of this entry.
- Updated the persistent agent prompt and goal so when CUDA is unavailable on this Mac, the loop can use the `uv`-based MLX proxy lane honestly instead of stalling or pretending record-track progress.
- Most likely next step: let the smoke finish, inspect the quantized roundtrip `val_bpb`, then queue an overnight frontier proxy run through `scripts/stage_mac_proxy_frontier_manifest.sh`.

## 2026-03-22 22:55 CDT - Durable Overnight Handoff On macOS

- Found and fixed a real controller bug in `research-agent/loopctl.py`: generated `job.sh` wrappers had leading spaces before the shebang, which caused `Exec format error` on manifest launches.
- Tightened the Mac frontier idle wrapper so it waits on the active `run_mlx_proxy_experiment.py` command line instead of a too-specific `train_gpt_mlx.py` path match.
- Added a direct detached launcher path plus a macOS LaunchAgent installer/status/uninstall set under `research-agent/` for the Mac frontier run.
- Installed the one-shot LaunchAgent `com.parameter-golf.mac-proxy-frontier` with `RUN_ID=thwu1_mlx_mac_frontier_20260322_225214_launchd`.
- Verified the LaunchAgent is loaded via `launchctl list` and its stdout log shows `waiting_for_idle`, so after the current smoke process disappears the Mac should promote automatically into the longer `uv`-backed frontier MLX run.
- Most likely next step: inspect `research-experiments/runs/thwu1_mlx_mac_frontier_20260322_225214_launchd/` and the candidate log tomorrow morning for the overnight frontier result, then decide whether to widen data or retune the local frontier env before promotion to H100s.

## 2026-03-23T03:49:06.412148+00:00 - Manifest Waiting For Runtime: rank1_mixed_qat_warmdown_ramp_seed42_20260323

- Path: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/rank1_mixed_qat_warmdown_ramp_seed42_20260323.json
- missing python module: torch
- missing python module: numpy
- missing python module: sentencepiece

## 2026-03-23T03:49:06.415145+00:00 - Manifest Ingested: thwu1_mlx_mac_frontier_20260322_224850

- From: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/thwu1_mlx_mac_frontier_20260322_224850.json
- To: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/pending/thwu1_mlx_mac_frontier_20260322_224850.json

## 2026-03-23T03:50:12.483875+00:00 - Manifest Ingested: thwu1_mlx_mac_frontier_20260322_224947

- From: /Users/aryan/Desktop/golf/research-experiments/manifests/pending/thwu1_mlx_mac_frontier_20260322_224947.json
- To: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/pending/thwu1_mlx_mac_frontier_20260322_224947.json

## 2026-03-23T03:50:12.488199+00:00 - Controller Job Launched: thwu1_mlx_mac_frontier_20260322_224947

- Description: Mac MLX frontier proxy run for the thwu1-derived candidate
- Job kind: proxy
- Command: bash scripts/run_mac_proxy_frontier_when_idle.sh
- CWD: /Users/aryan/Desktop/golf/research-experiments
- Manifest: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/running/thwu1_mlx_mac_frontier_20260322_224947.json
- Log: /Users/aryan/Desktop/golf/research-agent/loop/runtime/jobs/thwu1_mlx_mac_frontier_20260322_224947/run.log
- Timeout seconds: 32400
- PID: 48879

## 2026-03-23T03:52:13.583509+00:00 - Controller Job Result: thwu1_mlx_mac_frontier_20260322_224947

- Status: failed
- Job kind: proxy
- Exit code: 1
- Description: Mac MLX frontier proxy run for the thwu1-derived candidate
- Command: bash scripts/run_mac_proxy_frontier_when_idle.sh
- val_bpb: n/a
- artifact_bytes: n/a
- train_time_ms: n/a
- Manifest: /Users/aryan/Desktop/golf/research-agent/loop/runtime/queue/failed/thwu1_mlx_mac_frontier_20260322_224947.json
- Log: /Users/aryan/Desktop/golf/research-agent/loop/runtime/jobs/thwu1_mlx_mac_frontier_20260322_224947/run.log

## 2026-03-22 22:57 CDT - Mac Proxy uv Cache Fix And Proxy Restage

- Traced the failed `thwu1_mlx_mac_frontier_20260322_224947` proxy launch to `uv` trying to initialize `~/.cache/uv`, which is not writable in this workspace-controlled loop.
- Added `scripts/mac_proxy_uv_env.sh` and sourced it from `scripts/setup_mac_proxy_env.sh`, `scripts/prepare_mac_proxy_data.sh`, `scripts/run_mac_proxy_smoke.sh`, and `scripts/run_mac_proxy_frontier.sh` so all Mac proxy helpers now use the repo-local cache at `research-experiments/cache/uv`.
- Hardened `scripts/run_mac_proxy_frontier_when_idle.sh` so an unavailable `pgrep` process probe logs a one-line notice and launches immediately instead of failing the run wrapper.
- Updated `MAC_PROXY.md` to document the repo-local `uv` cache behavior and preserved the detached-launch helper that was already in the worktree.
- Verified the repaired proxy lane with `uv run ... scripts/run_mlx_proxy_experiment.py --preflight-only`; `runs/thwu1_mlx_mac_frontier_preflight_20260323/stats.json` reports `status=ready` against the official tokenizer and cached FineWeb shards.
- Parked the blocked CUDA-only manifest at `manifests/parked/rank1_mixed_qat_warmdown_ramp_seed42_20260323.json` and staged exactly one runnable Mac proxy manifest at `manifests/pending/thwu1_mlx_mac_frontier_20260323_0001.json` with `job_kind=proxy`.
- Most likely next step: let the controller ingest `thwu1_mlx_mac_frontier_20260323_0001`, inspect the first successful frontier proxy result, then port the mixed-QAT/export-aware record candidate changes into the MLX proxy lane before re-staging the CUDA experiment on a real multi-GPU runtime.
