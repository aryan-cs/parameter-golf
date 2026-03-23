#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    parser = argparse.ArgumentParser(description="Run a real Parameter Golf experiment candidate.")
    parser.add_argument("--experiment-dir", required=True, help="Path to a candidate directory containing train_gpt.py")
    parser.add_argument("--run-id", required=True, help="Run identifier used for logs and stats")
    parser.add_argument("--seed", type=int, default=42, help="Seed override passed to the train script")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="Number of local torchrun processes to launch")
    parser.add_argument("--stats-path", required=True, help="Where to write JSON metrics for the controller")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Token dataset directory")
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH), help="SentencePiece tokenizer model path")
    parser.add_argument("--preflight-only", action="store_true", help="Validate environment without launching training")
    return parser.parse_args()


def load_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "experiment.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


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
            [
                r"Total submission size int8\+zlib: ([0-9]+) bytes",
                r"Total submission size: ([0-9]+) bytes",
            ],
            int,
        ),
        "train_time_ms": last_match(text, [r"train_time:([0-9]+)ms"], int),
        "eval_time_ms": last_match(text, [r"eval_time:([0-9]+)ms"], int),
    }


def torch_probe() -> tuple[list[str], int]:
    issues: list[str] = []
    if importlib.util.find_spec("torch") is None:
        return ["missing python module: torch"], 0
    import torch  # type: ignore

    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
        return issues, 0
    return issues, int(torch.cuda.device_count())


def build_preflight(args: argparse.Namespace, experiment_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    entrypoint = experiment_dir / str(config.get("entrypoint", "train_gpt.py"))
    data_path = resolve_path(str(args.data_path), ROOT)
    tokenizer_path = resolve_path(str(args.tokenizer_path), ROOT)

    if not entrypoint.exists():
        issues.append(f"missing experiment entrypoint: {entrypoint}")

    required_modules = list(config.get("required_python_modules", ["torch", "numpy", "sentencepiece"]))
    for module_name in required_modules:
        if importlib.util.find_spec(str(module_name)) is None:
            issues.append(f"missing python module: {module_name}")

    torch_issues, cuda_devices = torch_probe()
    for issue in torch_issues:
        if issue not in issues:
            issues.append(issue)

    required_cuda_devices = int(config.get("required_cuda_devices", args.nproc_per_node))
    if cuda_devices < required_cuda_devices:
        issues.append(f"need {required_cuda_devices} CUDA devices, found {cuda_devices}")

    if not data_path.exists():
        issues.append(f"missing data path: {data_path}")
    else:
        train_shards = len(list(data_path.glob("fineweb_train_*.bin")))
        val_shards = len(list(data_path.glob("fineweb_val_*.bin")))
        if train_shards == 0:
            issues.append(f"no train shards found in {data_path}")
        if val_shards == 0:
            issues.append(f"no val shards found in {data_path}")

    if not tokenizer_path.exists():
        issues.append(f"missing tokenizer path: {tokenizer_path}")

    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "entrypoint": str(entrypoint),
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "cuda_device_count": cuda_devices,
        "required_cuda_devices": required_cuda_devices,
    }


def write_stats(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_command(args: argparse.Namespace, experiment_dir: Path, config: dict[str, Any]) -> tuple[list[str], dict[str, str], Path]:
    entrypoint = experiment_dir / str(config.get("entrypoint", "train_gpt.py"))
    env = os.environ.copy()
    for key, value in dict(config.get("env", {})).items():
        env[str(key)] = str(value)
    env["RUN_ID"] = args.run_id
    env["SEED"] = str(args.seed)
    env["DATA_PATH"] = str(resolve_path(str(args.data_path), ROOT))
    env["TOKENIZER_PATH"] = str(resolve_path(str(args.tokenizer_path), ROOT))

    if args.nproc_per_node > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            entrypoint.name,
        ]
    else:
        command = [sys.executable, entrypoint.name]
    return command, env, entrypoint


def main() -> int:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment_dir, ROOT)
    config = load_experiment_config(experiment_dir)
    stats_path = resolve_path(args.stats_path, ROOT)
    preflight = build_preflight(args, experiment_dir, config)

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

    command, env, entrypoint = build_command(args, experiment_dir, config)
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
        **metrics,
    }
    write_stats(stats_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
