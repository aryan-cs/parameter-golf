#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_URL = "https://github.com/openai/parameter-golf.git"
DEFAULT_CLONE_DIR = ROOT / "cache" / "openai-parameter-golf"
DEFAULT_SNAPSHOT_DIR = ROOT / "workbench" / "official_top_records"
DEFAULT_TRACK_DIR = "records/track_10min_16mb"


@dataclass(frozen=True)
class RecordEntry:
    rank: int
    folder: Path
    name: str
    val_bpb: float
    score_source: str
    submission: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone the official openai/parameter-golf repo, rank record-track submissions by "
            "best available score metadata, and snapshot the current top folders into the research-experiments workspace."
        )
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--clone-dir", default=str(DEFAULT_CLONE_DIR))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--track-dir", default=DEFAULT_TRACK_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt git pull when the clone already exists.",
    )
    return parser.parse_args()


def run_checked(args: list[str], cwd: Path | None = None, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=not allow_failure,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        if allow_failure:
            return exc
        if exc.stdout:
            print(exc.stdout, file=sys.stderr, end="" if exc.stdout.endswith("\n") else "\n")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="" if exc.stderr.endswith("\n") else "\n")
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_clone(repo_url: str, clone_dir: Path, refresh: bool) -> None:
    if clone_dir.exists():
        if not (clone_dir / ".git").exists():
            raise RuntimeError(f"Clone dir exists but is not a git repo: {clone_dir}")
        print(f"clone_status=reuse clone_dir={clone_dir}")
        if refresh:
            result = run_checked(["git", "pull", "--ff-only"], cwd=clone_dir, allow_failure=True)
            if result.returncode == 0:
                print("clone_refresh=ok")
            else:
                print("clone_refresh=failed_using_existing_clone", file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["git", "clone", "--depth", "1", repo_url, str(clone_dir)])
    print(f"clone_status=created clone_dir={clone_dir}")


def repo_head(clone_dir: Path) -> str:
    result = run_checked(["git", "rev-parse", "HEAD"], cwd=clone_dir)
    return result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_val_bpb(payload: dict[str, Any], readme_text: str | None) -> tuple[float | None, str | None]:
    for key in ("val_bpb", "best_val_bpb", "mean_val_bpb", "final_val_bpb", "score"):
        value = payload.get(key)
        parsed = coerce_float(value)
        if parsed is not None:
            return parsed, f"submission:{key}"

    if readme_text:
        patterns = (
            ("readme:mean_val_bpb", r"mean\s+val_bpb\s*[=:]\s*([0-9]+\.[0-9]+)"),
            ("readme:val_bpb", r"\bval_bpb\s*[=:]\s*([0-9]+\.[0-9]+)"),
        )
        for label, pattern in patterns:
            matches = re.findall(pattern, readme_text, flags=re.IGNORECASE)
            if matches:
                parsed = coerce_float(matches[0])
                if parsed is not None:
                    return parsed, label
    return None, None


def find_primary_train_script(folder: Path) -> Path | None:
    preferred_names = ("train_gpt.py", "train_gpt_v5.py", "train_gpt_mlx.py")
    for name in preferred_names:
        candidate = folder / name
        if candidate.exists():
            return candidate

    candidates = sorted(path for path in folder.glob("*.py") if path.name.startswith("train"))
    if candidates:
        return candidates[0]

    fallbacks = sorted(folder.glob("*.py"))
    if fallbacks:
        return fallbacks[0]
    return None


def load_ranked_records(track_dir: Path, top_k: int) -> list[RecordEntry]:
    entries: list[RecordEntry] = []
    skipped: list[tuple[Path, dict[str, Any]]] = []
    for submission_path in sorted(track_dir.glob("*/submission.json")):
        payload = load_json(submission_path)
        readme_text = read_text_if_exists(submission_path.parent / "README.md")
        val_bpb, score_source = extract_val_bpb(payload, readme_text)
        if val_bpb is None or score_source is None:
            skipped.append((submission_path, payload))
            continue
        entries.append(
            RecordEntry(
                rank=0,
                folder=submission_path.parent,
                name=str(payload.get("name") or submission_path.parent.name),
                val_bpb=val_bpb,
                score_source=score_source,
                submission=payload,
            )
        )
    for submission_path, payload in skipped:
        print(
            f"score_extract=missing folder={submission_path.parent.name} available_keys={','.join(sorted(payload))}",
            file=sys.stderr,
        )
    if not entries:
        raise RuntimeError(f"No ranked record submissions found under {track_dir}")
    ranked = sorted(entries, key=lambda entry: (entry.val_bpb, entry.folder.name))
    return [
        RecordEntry(
            rank=index,
            folder=entry.folder,
            name=entry.name,
            val_bpb=entry.val_bpb,
            score_source=entry.score_source,
            submission=entry.submission,
        )
        for index, entry in enumerate(ranked[:top_k], start=1)
    ]


def reset_snapshot_dir(snapshot_dir: Path) -> None:
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)


def snapshot_records(entries: list[RecordEntry], snapshot_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for entry in entries:
        target = snapshot_dir / f"rank_{entry.rank:02d}_{entry.folder.name}"
        shutil.copytree(entry.folder, target)
        readme_path = target / "README.md"
        train_script = find_primary_train_script(target)
        copied.append(
            {
                "rank": entry.rank,
                "name": entry.name,
                "val_bpb": entry.val_bpb,
                "score_source": entry.score_source,
                "source_dir": str(entry.folder),
                "snapshot_dir": str(target),
                "submission_path": str(entry.folder / "submission.json"),
                "train_script_path": None if train_script is None else str(train_script),
                "readme_path": str(readme_path) if readme_path.exists() else None,
                "submission": entry.submission,
            }
        )
    return copied


def write_summary(snapshot_dir: Path, payload: dict[str, Any]) -> None:
    summary_path = snapshot_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Official Top Record Snapshot",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Repo URL: `{payload['repo_url']}`",
        f"- Repo HEAD: `{payload['repo_head']}`",
        f"- Clone dir: `{payload['clone_dir']}`",
        f"- Track dir: `{payload['track_dir']}`",
        "",
        "## Ranked Seeds",
    ]
    for item in payload["top_records"]:
        lines.extend(
            [
                f"{item['rank']}. `{item['name']}`",
                f"   - `val_bpb={item['val_bpb']:.4f}`",
                f"   - score source: `{item['score_source']}`",
                f"   - source: `{item['source_dir']}`",
                f"   - snapshot: `{item['snapshot_dir']}`",
                f"   - train script: `{item['train_script_path']}`",
            ]
        )
    (snapshot_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_path={summary_path}")


def main() -> int:
    args = parse_args()
    clone_dir = Path(args.clone_dir).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()

    ensure_clone(args.repo_url, clone_dir, args.refresh)
    head = repo_head(clone_dir)
    track_dir = clone_dir / args.track_dir
    ranked = load_ranked_records(track_dir, args.top_k)

    reset_snapshot_dir(snapshot_dir)
    copied = snapshot_records(ranked, snapshot_dir)

    payload = {
        "generated_at": utc_now(),
        "repo_url": args.repo_url,
        "repo_head": head,
        "clone_dir": str(clone_dir),
        "track_dir": str(track_dir),
        "snapshot_dir": str(snapshot_dir),
        "top_records": copied,
    }
    write_summary(snapshot_dir, payload)

    for item in copied:
        print(
            f"top_record rank={item['rank']} val_bpb={item['val_bpb']:.4f} "
            f"name={item['name']} score_source={item['score_source']} "
            f"train_script={item['train_script_path']} snapshot_dir={item['snapshot_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
