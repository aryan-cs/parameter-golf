#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINAL_LINE_RE = re.compile(r"final_[^\s]* .*val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)(?: eval_time:(?P<eval_ms>[0-9.]+)ms)?")
STEP_VAL_RE = re.compile(r"step:(?P<step>\d+)/(?P<iters>\d+) val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+) train_time:(?P<train_ms>[0-9.]+)ms")
SERIALIZED_RE = re.compile(r"(Serialized model|serialized_model(?:_int8_zlib)?):[^0-9]*(?P<bytes>\d+) bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--train-script", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log_path = run_dir / "train.log"
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    final_matches = list(FINAL_LINE_RE.finditer(text))
    final_metrics = final_matches[-1].groupdict() if final_matches else {}
    step_matches = list(STEP_VAL_RE.finditer(text))
    last_step_metrics = step_matches[-1].groupdict() if step_matches else {}
    serialized_bytes = [int(m.group("bytes")) for m in SERIALIZED_RE.finditer(text)]

    artifacts = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.suffix in {".pt", ".ptz", ".npz", ".txt", ".json"}:
            artifacts[path.name] = path.stat().st_size

    summary = {
        "candidate": args.candidate,
        "seed": args.seed,
        "train_script": args.train_script,
        "commit": (run_dir / "commit.txt").read_text(encoding="utf-8").strip() if (run_dir / "commit.txt").exists() else "unknown",
        "final_metrics": final_metrics,
        "last_step_metrics": last_step_metrics,
        "serialized_bytes_from_log": serialized_bytes,
        "artifacts": artifacts,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    submission_template = {
        "candidate": args.candidate,
        "seed": args.seed,
        "commit": summary["commit"],
        "train_script": args.train_script,
        "artifact_files": artifacts,
        "final_val_bpb": final_metrics.get("val_bpb"),
    }
    (run_dir / "submission_template.json").write_text(
        json.dumps(submission_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
