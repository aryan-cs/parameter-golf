#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "m4-mini": [
        {"run_id": "m4_d192_l4", "D_MODEL": "192", "N_HEADS": "6", "D_FF": "512", "N_LOOPS": "4"},
        {"run_id": "m4_d256_l4", "D_MODEL": "256", "N_HEADS": "8", "D_FF": "682", "N_LOOPS": "4"},
        {"run_id": "m4_d256_l6", "D_MODEL": "256", "N_HEADS": "8", "D_FF": "682", "N_LOOPS": "6"},
        {"run_id": "m4_d320_l4", "D_MODEL": "320", "N_HEADS": "8", "D_FF": "853", "N_LOOPS": "4"},
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small ablation sweep over train_gpt.py configs.")
    parser.add_argument("--preset", default="m4-mini", choices=sorted(PRESETS), help="Named sweep preset.")
    parser.add_argument("--data-path", required=True, help="Packed train token directory.")
    parser.add_argument("--val-data-path", required=True, help="Packed val token directory.")
    parser.add_argument("--max-steps", type=int, default=30, help="Training steps per config.")
    parser.add_argument("--train-batch-tokens", type=int, default=8192, help="Batch token budget per step.")
    parser.add_argument("--val-batch-tokens", type=int, default=8192, help="Validation token budget.")
    parser.add_argument("--val-loss-every", type=int, default=10, help="Validation cadence.")
    parser.add_argument("--device", default="", help="Optional explicit device.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of preset configs to run.")
    parser.add_argument("--output", default="runs/m4-mini/results.jsonl", help="Where to append result summaries.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    summaries = []
    configs = PRESETS[args.preset][: args.limit] if args.limit > 0 else PRESETS[args.preset]

    for index, config in enumerate(configs, start=1):
        run_id = config["run_id"]
        stats_path = output_path.parent / f"{run_id}.json"
        env = os.environ.copy()
        env.update({key: value for key, value in config.items() if key != "run_id"})
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": args.data_path,
                "VAL_DATA_PATH": args.val_data_path,
                "MAX_STEPS": str(args.max_steps),
                "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
                "VAL_BATCH_TOKENS": str(args.val_batch_tokens),
                "VAL_LOSS_EVERY": str(args.val_loss_every),
                "STATS_PATH": str(stats_path),
            }
        )
        if args.device:
            env["DEVICE"] = args.device

        print(f"[{index}/{len(configs)}] run_id={run_id}", flush=True)
        subprocess.run([sys.executable, "train_gpt.py"], check=True, env=env)

        summary = json.loads(stats_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

    ranked = sorted(summaries, key=lambda item: item["final_val_bpb"])
    print("\n=== sweep_ranking ===")
    for rank, summary in enumerate(ranked, start=1):
        print(
            f"{rank}. {summary['run_id']} "
            f"final_val_bpb={summary['final_val_bpb']:.4f} "
            f"params={summary['parameters']} "
            f"artifact={summary['total_artifact_bytes']}"
        )
    print(f"results_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
