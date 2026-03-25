#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_submission_metadata import merge_summaries, parse_log


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(10):
        if (current / "README.md").is_file() and (current / "records").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise RuntimeError("could not locate repo root")


def find_train_script(folder: Path) -> Path:
    exact = folder / "train_gpt.py"
    if exact.is_file():
        return exact
    variants = sorted(folder.glob("train_gpt*.py"))
    if variants:
        return variants[0]
    raise FileNotFoundError(f"no train_gpt*.py found in {folder}")


def copy_file(src: Path, dst: Path) -> dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "bytes": dst.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean submission-style snapshot folder from a dev record folder."
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--log",
        dest="logs",
        type=Path,
        action="append",
        required=True,
        help="Log file to copy into the snapshot root. May be repeated.",
    )
    parser.add_argument(
        "--extra-file",
        dest="extra_files",
        type=Path,
        action="append",
        default=[],
        help="Extra file to copy into the snapshot root. May be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Replace output_dir if it already exists.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_submission.py on the finished snapshot.",
    )
    parser.add_argument("--name")
    parser.add_argument("--author")
    parser.add_argument("--github-id")
    parser.add_argument("--date")
    parser.add_argument("--blurb")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    validator_path = repo_root / "validate_submission.py"

    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, object]] = []

    readme_path = source_dir / "README.md"
    if not readme_path.is_file():
        raise SystemExit(f"missing README.md in {source_dir}")
    copied.append(copy_file(readme_path, output_dir / "README.md"))

    train_script = find_train_script(source_dir)
    copied.append(copy_file(train_script, output_dir / train_script.name))

    requirements_path = source_dir / "requirements.txt"
    if requirements_path.is_file():
        copied.append(copy_file(requirements_path, output_dir / "requirements.txt"))

    source_submission_path = source_dir / "submission.json"
    base_submission: dict[str, object] = {}
    if source_submission_path.is_file():
        base_submission = json.loads(source_submission_path.read_text(encoding="utf-8"))

    resolved_logs = []
    for log_path in args.logs:
        resolved = log_path if log_path.is_absolute() else (repo_root / log_path)
        if not resolved.is_file():
            raise SystemExit(f"log file not found: {resolved}")
        resolved_logs.append(resolved.resolve())

    summaries = [parse_log(log_path) for log_path in resolved_logs]
    merged = merge_summaries(summaries) if len(summaries) > 1 else summaries[0]

    submission = dict(base_submission)
    for key, value in (
        ("name", args.name),
        ("author", args.author),
        ("github_id", args.github_id),
        ("date", args.date),
    ):
        if value is not None:
            submission[key] = value
    if "submission_val_bpb" in merged:
        submission["val_bpb"] = merged["submission_val_bpb"]
    if "bytes_total" in merged and merged["bytes_total"] is not None:
        submission["bytes_total"] = merged["bytes_total"]
    if args.blurb is not None:
        submission["blurb"] = args.blurb
    else:
        metric = merged.get("submission_metric")
        val_bpb = submission.get("val_bpb")
        bytes_total = submission.get("bytes_total")
        metric_text = f"{metric}={val_bpb}" if metric is not None and val_bpb is not None else f"val_bpb={val_bpb}"
        bytes_text = f", bytes_total={bytes_total}" if bytes_total is not None else ""
        submission["blurb"] = f"Snapshot from {source_dir.name}: {metric_text}{bytes_text}."

    submission_path = output_dir / "submission.json"
    submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    copied.append({"src": str(source_submission_path), "dst": str(submission_path), "bytes": submission_path.stat().st_size})

    seen_names: set[str] = set()
    for src in resolved_logs + list(args.extra_files):
        resolved = src if src.is_absolute() else (repo_root / src)
        if not resolved.is_file():
            raise SystemExit(f"extra file not found: {resolved}")
        if resolved.name in seen_names:
            raise SystemExit(f"duplicate output filename requested: {resolved.name}")
        seen_names.add(resolved.name)
        copied.append(copy_file(resolved, output_dir / resolved.name))

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "submission_metric": merged.get("submission_metric"),
        "submission_val_bpb": merged.get("submission_val_bpb"),
        "bytes_total": submission.get("bytes_total"),
        "copied_files": copied,
    }
    manifest_path = output_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    print(json.dumps({
        "output_dir": str(output_dir),
        "submission_metric": merged.get("submission_metric"),
        "submission_val_bpb": merged.get("submission_val_bpb"),
        "bytes_total": submission.get("bytes_total"),
        "snapshot_total_bytes": total_bytes,
        "copied_count": len(copied),
    }, indent=2))

    if args.validate:
        subprocess.run([sys.executable, str(validator_path), str(output_dir)], check=True)


if __name__ == "__main__":
    main()
