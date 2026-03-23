#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "cache" / "openai-parameter-golf" / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = ROOT / "cache" / "openai-parameter-golf" / "data" / "tokenizers" / "fineweb_1024_bpe.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Apple Silicon MLX proxy experiment.")
    parser.add_argument("--experiment-dir", required=True, help="Candidate directory containing train_gpt_mlx.py")
    parser.add_argument("--run-id", required=True, help="Run identifier for logs and stats")
    parser.add_argument("--seed", type=int, default=42, help="Seed override")
    parser.add_argument("--stats-path", required=True, help="Where to write JSON metrics")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Official challenge dataset directory")
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH), help="SentencePiece model path")
    parser.add_argument("--set-env", action="append", default=[], help="Extra KEY=VALUE env overrides")
    parser.add_argument("--set-env-file", default=None, help="Path to JSON env overrides")
    parser.add_argument("--preflight-only", action="store_true", help="Validate environment without launching training")
    return parser.parse_args()


def resolve_path(raw_path: str, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def load_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.set_env_file:
        payload = json.loads(resolve_path(str(args.set_env_file), ROOT).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--set-env-file must contain a JSON object")
        for key, value in payload.items():
            overrides[str(key)] = str(value)
    for item in args.set_env:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --set-env, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def write_stats(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def last_match(text: str, patterns: list[str], cast) -> Any:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            try:
                return cast(matches[-1])
            except Exception:
                continue
    return None


def parse_train_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "val_bpb": None,
            "val_loss": None,
            "artifact_bytes": None,
            "train_time_ms": None,
            "eval_time_ms": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "val_bpb": last_match(
            text,
            [
                r"final_int8_zlib_roundtrip_exact val_loss:[0-9]+\.[0-9]+ val_bpb:([0-9]+\.[0-9]+)",
                r"final_int8_zlib_roundtrip val_loss:[0-9]+\.[0-9]+ val_bpb:([0-9]+\.[0-9]+)",
                r"\bval_bpb:([0-9]+\.[0-9]+)",
            ],
            float,
        ),
        "val_loss": last_match(
            text,
            [
                r"final_int8_zlib_roundtrip_exact val_loss:([0-9]+\.[0-9]+)",
                r"final_int8_zlib_roundtrip val_loss:([0-9]+\.[0-9]+)",
                r"\bval_loss:([0-9]+\.[0-9]+)",
            ],
            float,
        ),
        "artifact_bytes": last_match(
            text,
            [r"serialized_model_int8_zlib:([0-9]+) bytes"],
            int,
        ),
        "train_time_ms": last_match(text, [r"train_time:([0-9]+)ms"], int),
        "eval_time_ms": last_match(text, [r"eval_time:([0-9]+)ms"], int),
    }


def build_preflight(args: argparse.Namespace, experiment_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    entrypoint = experiment_dir / "train_gpt_mlx.py"
    data_path = resolve_path(str(args.data_path), ROOT)
    tokenizer_path = resolve_path(str(args.tokenizer_path), ROOT)
    env_overrides = load_env_overrides(args)

    if platform.system() != "Darwin":
        issues.append(f"expected Darwin for MLX proxy runs, found {platform.system()}")
    if platform.machine() != "arm64":
        issues.append(f"expected arm64 Apple Silicon, found {platform.machine()}")

    for module_name in ("mlx", "mlx.core", "numpy", "sentencepiece"):
        try:
            found = importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:
            found = False
        if not found:
            issues.append(f"missing python module: {module_name}")

    if not entrypoint.exists():
        issues.append(f"missing experiment entrypoint: {entrypoint}")
    if not data_path.exists():
        issues.append(f"missing data path: {data_path}")
    if not tokenizer_path.exists():
        issues.append(f"missing tokenizer path: {tokenizer_path}")

    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "entrypoint": str(entrypoint),
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "env_overrides": env_overrides,
        "platform": platform.platform(),
    }


def build_command(
    args: argparse.Namespace,
    experiment_dir: Path,
) -> tuple[list[str], dict[str, str], Path, dict[str, str]]:
    entrypoint = experiment_dir / "train_gpt_mlx.py"
    env = os.environ.copy()
    env_overrides = load_env_overrides(args)
    for key, value in env_overrides.items():
        env[str(key)] = str(value)
    env["RUN_ID"] = args.run_id
    env["SEED"] = str(args.seed)
    env["DATA_PATH"] = str(resolve_path(str(args.data_path), ROOT))
    env["TOKENIZER_PATH"] = str(resolve_path(str(args.tokenizer_path), ROOT))
    command = [sys.executable, entrypoint.name]
    return command, env, entrypoint, env_overrides


def main() -> int:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment_dir, ROOT)
    stats_path = resolve_path(args.stats_path, ROOT)
    preflight = build_preflight(args, experiment_dir)

    if args.preflight_only or not preflight["ready"]:
        payload = {
            "status": "ready" if preflight["ready"] else "preflight_failed",
            "run_id": args.run_id,
            "seed": args.seed,
            "experiment_dir": str(experiment_dir),
            "preflight": preflight,
        }
        write_stats(stats_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.preflight_only and preflight["ready"] else (2 if not preflight["ready"] else 0)

    command, env, entrypoint, env_overrides = build_command(args, experiment_dir)
    started_at = time.time()
    print(f"launch_command={' '.join(command)}")
    print(f"experiment_dir={experiment_dir}")
    print(f"entrypoint={entrypoint}")

    completed = subprocess.run(command, cwd=experiment_dir, env=env, check=False)
    log_path = experiment_dir / "logs" / f"{args.run_id}.txt"
    metrics = parse_train_log(log_path)
    payload = {
        "status": "done" if completed.returncode == 0 else "failed",
        "run_id": args.run_id,
        "seed": args.seed,
        "experiment_dir": str(experiment_dir),
        "entrypoint": str(entrypoint),
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_wallclock_seconds": time.time() - started_at,
        "log_path": str(log_path),
        "preflight": preflight,
        "env_overrides": env_overrides,
        **metrics,
    }
    write_stats(stats_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
