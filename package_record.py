#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a finished run into an official-style records/ folder."
    )
    parser.add_argument("--stats", required=True, help="Path to a run stats JSON file.")
    parser.add_argument("--log", required=True, help="Path to the corresponding train log.")
    parser.add_argument("--name", required=True, help="Human-readable run name.")
    parser.add_argument("--slug", required=True, help="Filesystem-safe run slug.")
    parser.add_argument("--author", required=True, help="Submission author name.")
    parser.add_argument("--github-id", required=True, help="GitHub username.")
    parser.add_argument("--blurb", required=True, help="Short submission description.")
    parser.add_argument(
        "--track-dir",
        default="track_non_record_16mb",
        choices=["track_10min_16mb", "track_non_record_16mb"],
        help="Target records subdirectory.",
    )
    parser.add_argument(
        "--submission-track",
        default="",
        help="Optional submission.json track field, for example non-record-unlimited-compute-16mb.",
    )
    parser.add_argument(
        "--output-root",
        default="records",
        help="Root records directory to write into.",
    )
    parser.add_argument(
        "--train-script",
        default="train_gpt.py",
        help="Path to the train_gpt.py snapshot to copy into the record folder.",
    )
    parser.add_argument(
        "--readme-notes",
        default="",
        help="Optional extra note to append to the generated README.",
    )
    parser.add_argument(
        "--allow-over-budget",
        action="store_true",
        help="Allow packaging runs that exceed 16,000,000 bytes.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_dir_name(slug: str) -> str:
    return f"{datetime.now(timezone.utc).date().isoformat()}_{slug}"


def build_submission(stats: dict, args: argparse.Namespace) -> dict:
    payload = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb,
        "date": utc_now_iso(),
        "val_loss": stats["final_val_loss"],
        "val_bpb": stats["final_val_bpb"],
        "bytes_total": stats["total_artifact_bytes"],
        "bytes_code": stats["code_size_bytes"],
    }
    if args.submission_track:
        payload["track"] = args.submission_track
    if "best_val_loss" in stats:
        payload["best_val_loss"] = stats["best_val_loss"]
    if "best_val_bpb" in stats:
        payload["best_val_bpb"] = stats["best_val_bpb"]
    payload["bytes_model_int8_zlib"] = stats["compressed_model_size_bytes"]
    payload["steps"] = stats["steps"]
    payload["wallclock_seconds"] = stats["seconds"]
    return payload


def build_readme(stats: dict, args: argparse.Namespace, submission: dict) -> str:
    cfg = stats.get("config", {})
    lines = [
        f"This record captures `{args.name}`.",
        "",
        "Generated metadata:",
        f"- Track directory: `{args.track_dir}`",
        f"- Run id: `{stats.get('run_id', '<unknown>')}`",
        f"- Device: `{stats.get('device', '<unknown>')}`",
        f"- Final val_bpb: `{stats['final_val_bpb']:.8f}`",
        f"- Final val_loss: `{stats['final_val_loss']:.8f}`",
        f"- Total artifact bytes: `{stats['total_artifact_bytes']}`",
        f"- Compressed model bytes: `{stats['compressed_model_size_bytes']}`",
        f"- Code bytes: `{stats['code_size_bytes']}`",
        f"- Artifact budget ok: `{stats['artifact_budget_ok']}`",
        "",
        "Configuration:",
        f"- vocab_size: `{stats.get('vocab_size')}`",
        f"- token_dtype: `{stats.get('token_dtype')}`",
        f"- avg_bytes_per_token: `{stats.get('avg_bytes_per_token')}`",
        f"- d_model: `{cfg.get('d_model')}`",
        f"- n_heads: `{cfg.get('n_heads')}`",
        f"- d_ff: `{cfg.get('d_ff')}`",
        f"- n_loops: `{cfg.get('n_loops')}`",
        f"- seq_len: `{cfg.get('seq_len')}`",
        f"- train_batch_tokens: `{cfg.get('train_batch_tokens')}`",
        f"- val_batch_tokens: `{cfg.get('val_batch_tokens')}`",
        f"- max_steps: `{cfg.get('max_steps')}`",
        f"- data_path: `{cfg.get('data_path')}`",
        f"- val_data_path: `{cfg.get('val_data_path')}`",
        "",
        "Included files:",
        "- `README.md`",
        "- `submission.json`",
        "- `train.log`",
        "- `train_gpt.py`",
        "",
        "Notes:",
        "- This README was generated locally and should be edited before any public submission.",
        "- For a main-track PR, make sure the run is competition-valid under the official 8xH100 and full-validation requirements.",
    ]
    if args.readme_notes:
        lines.extend(["", args.readme_notes.strip()])
    lines.extend(
        [
            "",
            "submission.json preview:",
            "```json",
            json.dumps(submission, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    stats_path = Path(args.stats)
    log_path = Path(args.log)
    train_script_path = Path(args.train_script)

    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not train_script_path.is_file():
        raise FileNotFoundError(f"train_gpt.py snapshot not found: {train_script_path}")

    stats = load_json(stats_path)
    if not args.allow_over_budget and int(stats["total_artifact_bytes"]) > 16_000_000:
        raise ValueError(
            f"Run exceeds the 16,000,000 byte cap: total_artifact_bytes={stats['total_artifact_bytes']}"
        )

    root = Path(args.output_root) / args.track_dir
    root.mkdir(parents=True, exist_ok=True)
    target = root / record_dir_name(slugify(args.slug))
    if target.exists():
        raise FileExistsError(f"Target record directory already exists: {target}")
    target.mkdir(parents=True)

    submission = build_submission(stats, args)
    (target / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(build_readme(stats, args, submission), encoding="utf-8")
    shutil.copy2(log_path, target / "train.log")
    shutil.copy2(train_script_path, target / "train_gpt.py")

    print(f"record_dir={target}")
    print(f"submission_path={target / 'submission.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
