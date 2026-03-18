#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BUDGET_BYTES = 16_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a width-frontier training loop with persistent logs and summaries."
    )
    parser.add_argument("--data-path", required=True, help="Packed train token directory.")
    parser.add_argument("--val-data-path", required=True, help="Packed val token directory.")
    parser.add_argument("--widths", nargs="+", type=int, required=True, help="Model widths to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory for logs and result summaries.")
    parser.add_argument("--run-prefix", default="autopilot", help="Prefix for generated run ids.")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention head count.")
    parser.add_argument("--n-loops", type=int, default=4, help="Loop depth.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument(
        "--d-ff-scale",
        type=float,
        default=8.0 / 3.0,
        help="Feed-forward width multiplier relative to d_model.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Training steps per config.")
    parser.add_argument("--train-batch-tokens", type=int, default=8192, help="Train batch token budget.")
    parser.add_argument("--val-batch-tokens", type=int, default=8192, help="Validation batch token budget.")
    parser.add_argument("--val-loss-every", type=int, default=10, help="Validation cadence.")
    parser.add_argument("--device", default="", help="Optional explicit device.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing summaries when available.")
    return parser.parse_args()


def infer_d_ff(d_model: int, scale: float) -> int:
    return max(4, int(d_model * scale))


def build_run_id(prefix: str, width: int, loops: int) -> str:
    return f"{prefix}_d{width}_l{loops}"


def stream_process(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        log_handle.write("\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_ranking(title: str, summaries: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not summaries:
        print("no runs")
        return
    for rank, summary in enumerate(sorted(summaries, key=lambda item: item["final_val_bpb"]), start=1):
        print(
            f"{rank}. {summary['run_id']} "
            f"final_val_bpb={summary['final_val_bpb']:.4f} "
            f"params={summary['parameters']} "
            f"artifact={summary['total_artifact_bytes']}"
        )


def write_results(output_dir: Path, summaries: list[dict]) -> None:
    results_path = output_dir / "results.jsonl"
    leaderboard_path = output_dir / "leaderboard.json"
    ranked = sorted(summaries, key=lambda item: item["final_val_bpb"])
    with results_path.open("w", encoding="utf-8") as handle:
        for summary in ranked:
            handle.write(json.dumps(summary) + "\n")
    leaderboard = {
        "generated_at_unix": time.time(),
        "best_overall": ranked[0] if ranked else None,
        "best_under_budget": next((item for item in ranked if item["artifact_budget_ok"]), None),
        "runs": ranked,
    }
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2) + "\n", encoding="utf-8")
    print(f"results_path={results_path}")
    print(f"leaderboard_path={leaderboard_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    widths = list(dict.fromkeys(args.widths))

    for index, width in enumerate(widths, start=1):
        run_id = build_run_id(args.run_prefix, width, args.n_loops)
        d_ff = infer_d_ff(width, args.d_ff_scale)
        stats_path = output_dir / f"{run_id}.json"
        log_path = output_dir / f"{run_id}.log"

        if args.resume and stats_path.exists():
            print(f"[{index}/{len(widths)}] run_id={run_id} resume=summary", flush=True)
            summaries.append(load_summary(stats_path))
            continue

        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": args.data_path,
                "VAL_DATA_PATH": args.val_data_path,
                "D_MODEL": str(width),
                "N_HEADS": str(args.n_heads),
                "D_FF": str(d_ff),
                "N_LOOPS": str(args.n_loops),
                "SEQ_LEN": str(args.seq_len),
                "MAX_STEPS": str(args.max_steps),
                "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
                "VAL_BATCH_TOKENS": str(args.val_batch_tokens),
                "VAL_LOSS_EVERY": str(args.val_loss_every),
                "SEED": str(args.seed),
                "STATS_PATH": str(stats_path),
            }
        )
        if args.device:
            env["DEVICE"] = args.device

        command = [sys.executable, "train_gpt.py"]
        print(f"[{index}/{len(widths)}] run_id={run_id} d_ff={d_ff}", flush=True)
        stream_process(command, env, log_path)
        summary = load_summary(stats_path)
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print_ranking("autopilot_ranking", summaries)
    under_budget = [summary for summary in summaries if summary["artifact_budget_ok"]]
    print_ranking("autopilot_under_budget_ranking", under_budget)
    write_results(output_dir, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
