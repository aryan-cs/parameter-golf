#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CLONE = ROOT / "cache" / "openai-parameter-golf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official published challenge data into the cached official clone.")
    parser.add_argument("--variant", default="sp1024", help="Tokenizer/data variant to download")
    parser.add_argument("--train-shards", type=int, default=80, help="How many train shards to fetch")
    parser.add_argument("--repo-id", default=None, help="Optional HF repo override for cached_challenge_fineweb.py")
    parser.add_argument("--remote-root", default=None, help="Optional remote root override for cached_challenge_fineweb.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
