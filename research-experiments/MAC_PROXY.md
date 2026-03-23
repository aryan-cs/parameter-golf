# Mac Proxy Workflow

This workspace includes a Mac-native proxy lane for Apple Silicon using `uv` and MLX.

Use this path to validate architecture changes, training stability, export logic, and relative score movement on the official challenge data. These runs are proxy/dev runs, not real leaderboard submissions.

## Setup

```bash
bash research-experiments/scripts/setup_mac_proxy_env.sh
```

## Download a Small Official Data Subset

```bash
TRAIN_SHARDS=1 bash research-experiments/scripts/prepare_mac_proxy_data.sh
```

Increase `TRAIN_SHARDS` later when you want a stronger local proxy.

## Run a Fast Smoke Test

```bash
bash research-experiments/scripts/run_mac_proxy_smoke.sh
```

## Run a Longer Proxy Experiment

```bash
bash research-experiments/scripts/run_mac_proxy_frontier.sh
```

## Queue An Overnight Proxy Run Through The Controller

```bash
bash research-experiments/scripts/stage_mac_proxy_frontier_manifest.sh
bash research-agent/start_loop.sh
bash research-agent/status_loop.sh
```

This queues the longer MLX proxy run into the existing controller instead of keeping it as a one-off terminal process.

## Current Proxy Candidate

- `research-experiments/mac_proxy_candidates/2026-03-23_thwu1_mlx_proxy/`
- source: mirrored `thwu1/parameter-golf` MLX script
- goal: get a reliable local Apple Silicon experiment path running before promoting ideas to H100 runs
