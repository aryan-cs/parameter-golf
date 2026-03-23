#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CLONE = ROOT / "cache" / "openai-parameter-golf"
DEFAULT_REPO_URL = "https://github.com/openai/parameter-golf.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official published challenge data into the cached official clone.")
    parser.add_argument("--variant", default="sp1024", help="Tokenizer/data variant to download")
    parser.add_argument("--train-shards", type=int, default=80, help="How many train shards to fetch")
    parser.add_argument("--repo-id", default=None, help="Optional HF repo override for cached_challenge_fineweb.py")
    parser.add_argument("--remote-root", default=None, help="Optional remote root override for cached_challenge_fineweb.py")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL for the official openai/parameter-golf repo")
    parser.add_argument(
        "--refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refresh the cached official clone when it already exists.",
    )
    return parser.parse_args()


def run_checked(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)


def ensure_official_clone(repo_url: str, clone_dir: Path, refresh: bool) -> None:
    if clone_dir.exists():
        if not (clone_dir / ".git").exists():
            raise RuntimeError(f"Official cache dir exists but is not a git repo: {clone_dir}")
        print(f"official_clone_status=reuse clone_dir={clone_dir}")
        if refresh:
            print("official_clone_refresh=fetch")
            run_checked(["git", "fetch", "origin"], cwd=clone_dir)
            run_checked(["git", "reset", "--hard", "origin/HEAD"], cwd=clone_dir)
        return

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"official_clone_status=clone repo_url={repo_url} clone_dir={clone_dir}")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)


def main() -> int:
    args = parse_args()
    ensure_official_clone(args.repo_url, OFFICIAL_CLONE, args.refresh)
    script_path = OFFICIAL_CLONE / "data" / "cached_challenge_fineweb.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing helper script: {script_path}")

    env = os.environ.copy()
    if args.repo_id:
        env["MATCHED_FINEWEB_REPO_ID"] = args.repo_id
    if args.remote_root:
        env["MATCHED_FINEWEB_REMOTE_ROOT_PREFIX"] = args.remote_root

    command = [
        sys.executable,
        "data/cached_challenge_fineweb.py",
        "--variant",
        args.variant,
        "--train-shards",
        str(args.train_shards),
    ]
    print(f"launch_command={' '.join(command)}")
    completed = subprocess.run(command, cwd=OFFICIAL_CLONE, env=env, check=False)
    dataset_dir = OFFICIAL_CLONE / "data" / "datasets" / f"fineweb10B_{args.variant}"
    tokenizer_dir = OFFICIAL_CLONE / "data" / "tokenizers"
    print(f"dataset_dir={dataset_dir}")
    print(f"tokenizer_dir={tokenizer_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
