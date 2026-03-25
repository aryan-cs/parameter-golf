#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from prepare_submission_metadata import merge_summaries, parse_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a set of Parameter Golf run logs.")
    parser.add_argument("log_paths", nargs="+", type=Path)
    parser.add_argument(
        "--merge-logs",
        action="store_true",
        help="Treat all provided logs as parts of one logical run and merge metrics across them.",
    )
    args = parser.parse_args()

    runs = []
    metric_values = []
    metric_name = None
    byte_values = []

    if args.merge_logs:
        summaries = [parse_log(path) for path in args.log_paths]
        runs = [merge_summaries(summaries)]
    else:
        for path in args.log_paths:
            runs.append(parse_log(path))

    for summary in runs:
        if summary.get("submission_val_bpb") is not None:
            metric_values.append(float(summary["submission_val_bpb"]))
            metric_name = str(summary.get("submission_metric", "submission_val_bpb"))
        elif summary.get("last_val_bpb") is not None:
            metric_values.append(float(summary["last_val_bpb"]))
            metric_name = "last_val_bpb"
        if summary.get("bytes_total") is not None:
            byte_values.append(int(summary["bytes_total"]))

    output: dict[str, object] = {
        "num_runs": len(runs),
        "metric_name": metric_name,
        "runs": runs,
    }
    if metric_values:
        output["metric_mean"] = statistics.mean(metric_values)
        output["metric_min"] = min(metric_values)
        output["metric_max"] = max(metric_values)
        if len(metric_values) > 1:
            output["metric_sample_std"] = statistics.stdev(metric_values)
    if byte_values:
        output["bytes_min"] = min(byte_values)
        output["bytes_max"] = max(byte_values)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
