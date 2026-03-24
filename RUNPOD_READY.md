# Runpod Ready

## Current Best

- Best completed quantized score: `1.1159 val_bpb`
- Status: invalid
- Reason: artifact was `16,333,801` bytes, which is `333,801` bytes over the `16,000,000` byte cap

## Current Lane

- Primary recipe: `VRL + Full GPTQ`
- Best pre-export curve:
  - `1000 -> 1.3111`
  - `2000 -> 1.2527`
  - `3000 -> 1.2294`
  - `4000 -> 1.2136`
  - `5000 -> 1.1893`
  - `6000 -> 1.1636`
  - `6926 -> 1.1370`
- Byte-cap strategy:
  - packed int6 payloads
  - compact metadata
  - export-only prune ladder `05 -> 08 -> 11 -> 14 -> 17 -> 20`

## When Credits Land

Run this first to verify the local restart plan is still consistent:

```bash
python3 runpod/check_ready.py
```

### 1x H100 SXM recovery

Use this first if we want to verify the new serializer and prune ladder before spending on `8x`:

```bash
bash runpod/local_recover_export_chain.sh root@HOST /workspace/golf PORT 80
```

This will:
- sync the repo
- bootstrap the pod
- launch `non_ttt_vrl_gptq_1gpu_long`
- queue export-only `prune05 -> prune08 -> prune11 -> prune14 -> prune17 -> prune20`

### 8x H100 SXM recovery

Use this once we want the real final lane:

```bash
bash runpod/local_recover_export_chain_8gpu.sh root@HOST /workspace/golf PORT 80
```

This will:
- sync the repo
- bootstrap the pod
- launch `non_ttt_vrl_gptq_8gpu`
- queue export-only `prune05 -> prune08 -> prune11 -> prune14 -> prune17 -> prune20`

## After Launch

Watch the latest run:

```bash
bash runpod/local_watch_latest.sh root@HOST /workspace/golf PORT non_ttt_vrl_gptq 1337
```

Fetch results back:

```bash
bash runpod/local_fetch_from_pod.sh root@HOST /workspace/golf runs PORT
```

## Files To Check

- [JOURNAL.md](/Users/aryan/Desktop/golf/JOURNAL.md)
- [candidates/non_ttt_vrl_gptq/train_gpt.py](/Users/aryan/Desktop/golf/candidates/non_ttt_vrl_gptq/train_gpt.py)
- [runpod/local_recover_export_chain.sh](/Users/aryan/Desktop/golf/runpod/local_recover_export_chain.sh)
- [runpod/local_recover_export_chain_8gpu.sh](/Users/aryan/Desktop/golf/runpod/local_recover_export_chain_8gpu.sh)
- [runpod/pod_launch_export_chain.sh](/Users/aryan/Desktop/golf/runpod/pod_launch_export_chain.sh)
