#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_EXACT_RE = re.compile(
    r"(?P<metric>final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int6_sliding_window_s64_exact|legal_ttt_exact)"
    r"\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
BYTES_RE = re.compile(r"Total submission size int6\+lzma:\s+(?P<bytes>\d+)\s+bytes")


def parse_log(log_path: Path) -> dict:
    summary: dict[str, object] = {
        "log_path": str(log_path),
        "last_step": None,
        "iterations": None,
        "last_val_loss": None,
        "last_val_bpb": None,
        "bytes_total": None,
    }
    finals: dict[str, dict[str, float]] = {}

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if match := STEP_VAL_RE.search(line):
            summary["last_step"] = int(match.group("step"))
            summary["iterations"] = int(match.group("iterations"))
            summary["last_val_loss"] = float(match.group("val_loss"))
            summary["last_val_bpb"] = float(match.group("val_bpb"))
        if match := FINAL_EXACT_RE.search(line):
            finals[match.group("metric")] = {
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
            }
        if match := BYTES_RE.search(line):
            summary["bytes_total"] = int(match.group("bytes"))

    if finals:
        summary["final_metrics"] = finals
        if "legal_ttt_exact" in finals:
            summary["submission_metric"] = "legal_ttt_exact"
            summary["submission_val_bpb"] = finals["legal_ttt_exact"]["val_bpb"]
        elif "final_int6_sliding_window_exact" in finals:
            summary["submission_metric"] = "final_int6_sliding_window_exact"
            summary["submission_val_bpb"] = finals["final_int6_sliding_window_exact"]["val_bpb"]
        elif "final_int6_roundtrip_exact" in finals:
            summary["submission_metric"] = "final_int6_roundtrip_exact"
            summary["submission_val_bpb"] = finals["final_int6_roundtrip_exact"]["val_bpb"]
    return summary


def merge_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("at least one summary is required")

    merged: dict[str, object] = {
        "log_paths": [summary["log_path"] for summary in summaries],
        "last_step": None,
        "iterations": None,
        "last_val_loss": None,
        "last_val_bpb": None,
        "bytes_total": None,
    }
    final_metrics: dict[str, dict[str, float]] = {}

    for summary in summaries:
        for key in ("last_step", "iterations", "last_val_loss", "last_val_bpb", "bytes_total"):
            value = summary.get(key)
            if value is not None:
                merged[key] = value
        for metric_name, metric_payload in summary.get("final_metrics", {}).items():
            final_metrics[metric_name] = metric_payload

    if final_metrics:
        merged["final_metrics"] = final_metrics
        if "legal_ttt_exact" in final_metrics:
            merged["submission_metric"] = "legal_ttt_exact"
            merged["submission_val_bpb"] = final_metrics["legal_ttt_exact"]["val_bpb"]
        elif "final_int6_sliding_window_exact" in final_metrics:
            merged["submission_metric"] = "final_int6_sliding_window_exact"
            merged["submission_val_bpb"] = final_metrics["final_int6_sliding_window_exact"]["val_bpb"]
        elif "final_int6_roundtrip_exact" in final_metrics:
            merged["submission_metric"] = "final_int6_roundtrip_exact"
            merged["submission_val_bpb"] = final_metrics["final_int6_roundtrip_exact"]["val_bpb"]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract submission-ready metadata from a Parameter Golf run log.")
    parser.add_argument("log_paths", type=Path, nargs="+")
    parser.add_argument("--name", default="TODO")
    parser.add_argument("--author", default="TODO")
    parser.add_argument("--github-id", default="TODO")
    parser.add_argument("--date", default="TODO")
    parser.add_argument("--blurb", default="TODO")
    parser.add_argument("--write-submission-json", action="store_true")
    args = parser.parse_args()

    summaries = [parse_log(log_path) for log_path in args.log_paths]
    summary = merge_summaries(summaries) if len(summaries) > 1 else summaries[0]
    output = {
        "name": args.name,
        "author": args.author,
        "github_id": args.github_id,
        "date": args.date,
        "blurb": args.blurb,
        **summary,
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    if args.write_submission_json:
        submission = {
            "name": args.name,
            "author": args.author,
            "github_id": args.github_id,
            "date": args.date,
            "blurb": args.blurb,
            "val_bpb": output.get("submission_val_bpb"),
            "bytes_total": output.get("bytes_total"),
        }
        submission_path = args.log_paths[0].parent.parent / "submission.json"
        submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
