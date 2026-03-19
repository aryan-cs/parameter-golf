#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a serial queue of train_gpt.py experiments from a JSON manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a JSON file containing an `experiments` list.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated logs, copied stats, and leaderboard files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have a completed stats file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved queue without running it.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_stats_path(output_dir: Path, experiment: dict) -> Path:
    if "stats_path" in experiment:
        return Path(experiment["stats_path"])
    run_id = experiment["run_id"]
    return output_dir / f"{run_id}.json"


def resolve_log_path(output_dir: Path, experiment: dict) -> Path:
    if "log_path" in experiment:
        return Path(experiment["log_path"])
    run_id = experiment["run_id"]
    return output_dir / f"{run_id}.log"


def stream_process(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.flush()

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
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_ranking(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("no completed runs")
        return
    ranked = sorted(rows, key=lambda row: row["final_val_bpb"])
    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx}. {row['run_id']} "
            f"final_val_bpb={row['final_val_bpb']:.4f} "
            f"steps={row.get('steps', 'na')} "
            f"artifact={row.get('total_artifact_bytes', 'na')}"
        )


def write_leaderboard(output_dir: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: row["final_val_bpb"])
    leaderboard = {
        "generated_at_unix": time.time(),
        "best_overall": ranked[0] if ranked else None,
        "runs": ranked,
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    experiments = manifest.get("experiments", [])
    if not experiments:
        raise SystemExit("manifest has no experiments")

    if args.dry_run:
        for idx, experiment in enumerate(experiments, start=1):
            print(f"[{idx}/{len(experiments)}] {experiment['run_id']}")
            print(json.dumps(experiment, indent=2))
        return 0

    summaries: list[dict] = []

    for idx, experiment in enumerate(experiments, start=1):
        run_id = experiment["run_id"]
        stats_path = resolve_stats_path(output_dir, experiment)
        log_path = resolve_log_path(output_dir, experiment)

        if args.resume and stats_path.exists():
            print(f"[{idx}/{len(experiments)}] run_id={run_id} resume=summary", flush=True)
            summary = load_summary(stats_path)
            summary["log_path"] = str(log_path)
            summaries.append(summary)
            continue

        env = os.environ.copy()
        env.update({key: str(value) for key, value in experiment.get("env", {}).items()})
        env["RUN_ID"] = run_id
        env["STATS_PATH"] = str(stats_path)

        command = experiment.get("command", [sys.executable, "train_gpt.py"])
        print(f"[{idx}/{len(experiments)}] run_id={run_id}", flush=True)
        stream_process(command, env, log_path)

        summary = load_summary(stats_path)
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print_ranking("queue_ranking", summaries)
    write_leaderboard(output_dir, summaries)
    print(f"results_path={output_dir / 'results.jsonl'}")
    print(f"leaderboard_path={output_dir / 'leaderboard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
