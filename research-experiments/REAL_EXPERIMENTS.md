# Real Experiment Flow

This repo now has a real record-track candidate and a real training wrapper. The current lead candidate is:

- `record_candidates/2026-03-23_rank1_mixed_qat/`

Important runtime note:

- the macOS `launchd` loop on this machine is useful for local bookkeeping, but real leaderboard training must run in the actual GPU runtime
- if you want Colab-backed or other remote-GPU experiments, start the controller from that runtime, not from the local Mac background service

## Minimum Real-Run Setup

Install dependencies in the runtime that will actually execute training:

```bash
pip install -r research-experiments/cache/openai-parameter-golf/requirements.txt zstandard
```

Download the published challenge data:

```bash
python3 research-experiments/scripts/prepare_challenge_data.py --variant sp1024 --train-shards 80
```

Run preflight on the current candidate:

```bash
python3 research-experiments/scripts/run_record_experiment.py \
  --experiment-dir research-experiments/record_candidates/2026-03-23_rank1_mixed_qat \
  --run-id rank1_mixed_qat_seed42_20260323 \
  --seed 42 \
  --nproc-per-node 8 \
  --stats-path research-experiments/runs/rank1_mixed_qat_seed42_20260323/stats.json \
  --preflight-only
```

If preflight is clean, the same command without `--preflight-only` performs a real run and writes `stats.json` for the controller to ingest.

## Autonomous Mode In A Real GPU Runtime

From the GPU runtime itself:

```bash
python3 research-agent/loopctl.py start --config research-agent/loop/config.json
```

The controller ingests staged manifests from:

```bash
research-experiments/manifests/pending/
```

The current staged real manifest is:

- `research-experiments/manifests/pending/rank1_mixed_qat_seed42_20260323.json`
