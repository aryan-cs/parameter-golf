#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


EXACT_METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
MODEL_BYTES_RE = re.compile(r"serialized_model_int8_zlib:(?P<bytes>\d+) bytes")
MODEL_PARAMS_RE = re.compile(
    r"model_params:(?P<params>\d+) vocab_size:(?P<vocab>\d+) "
    r"layers:(?P<layers>\d+) dim:(?P<dim>\d+) heads:(?P<heads>\d+) kv_heads:(?P<kv_heads>\d+)"
)
STEP_RE = re.compile(r"step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>[-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the official train_gpt_mlx.py runner and save a parsed summary."
    )
    parser.add_argument("--official-root", required=True, help="Path to the cloned official parameter-golf repo.")
    parser.add_argument("--run-id", required=True, help="Run id forwarded to the official script.")
    parser.add_argument("--output-dir", required=True, help="Directory for the official script log and parsed JSON.")
    parser.add_argument("--data-path", required=True, help="Directory containing fineweb_train_*.bin and fineweb_val_*.bin.")
    parser.add_argument("--tokenizer-path", required=True, help="SentencePiece tokenizer .model file.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for the official script.")
    parser.add_argument("--resume", action="store_true", help="If the run log already exists, skip rerunning and only parse it.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment overrides forwarded to the official script.",
    )
    return parser.parse_args()


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Environment override is missing a key: {item}")
        overrides[key] = value
    return overrides


def parse_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8")
    summary: dict[str, object] = {
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size,
    }

    metric_matches = EXACT_METRIC_RE.findall(text)
    if metric_matches:
        val_loss, val_bpb = metric_matches[-1]
        summary["final_val_loss"] = float(val_loss)
        summary["final_val_bpb"] = float(val_bpb)

    model_bytes_matches = MODEL_BYTES_RE.findall(text)
    if model_bytes_matches:
        summary["bytes_model_int8_zlib"] = int(model_bytes_matches[-1])

    model_match = MODEL_PARAMS_RE.search(text)
    if model_match:
        summary["model_params"] = int(model_match.group("params"))
        summary["vocab_size"] = int(model_match.group("vocab"))
        summary["num_layers"] = int(model_match.group("layers"))
        summary["model_dim"] = int(model_match.group("dim"))
        summary["num_heads"] = int(model_match.group("heads"))
        summary["num_kv_heads"] = int(model_match.group("kv_heads"))

    step_matches = list(STEP_RE.finditer(text))
    if step_matches:
        last_step = step_matches[-1]
        summary["last_logged_step"] = int(last_step.group("step"))
        summary["target_iterations"] = int(last_step.group("iterations"))
        summary["last_logged_train_loss"] = float(last_step.group("train_loss"))

    return summary


def stream_process(command: list[str], env: dict[str, str]) -> None:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    args = parse_args()
    official_root = Path(args.official_root).resolve()
    script_path = official_root / "train_gpt_mlx.py"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.run_id}.txt"
    summary_path = output_dir / f"{args.run_id}.json"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": args.run_id,
            "OUT_DIR": str(output_dir),
            "DATA_PATH": str(Path(args.data_path).resolve()),
            "TOKENIZER_PATH": str(Path(args.tokenizer_path).resolve()),
        }
    )
    env.update(parse_env_overrides(args.env))

    command = [args.python, str(script_path)]
    launched_at = time.time()

    if args.resume and log_path.exists():
        print(f"resume=log run_id={args.run_id} log_path={log_path}")
    else:
        print(f"run_id={args.run_id}")
        print(f"log_path={log_path}")
        print(f"command={' '.join(command)}")
        print(f"output_dir={output_dir}")
        stream_process(command, env)

    if not log_path.exists():
        raise FileNotFoundError(f"Expected official log file was not created: {log_path}")

    summary = parse_log(log_path)
    summary.update(
        {
            "run_id": args.run_id,
            "official_root": str(official_root),
            "data_path": str(Path(args.data_path).resolve()),
            "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
            "output_dir": str(output_dir),
            "python": args.python,
            "env_overrides": parse_env_overrides(args.env),
            "completed_at_unix": time.time(),
            "launcher_wallclock_seconds": time.time() - launched_at,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"summary_path={summary_path}")
    if "final_val_bpb" in summary:
        print(
            "final_exact "
            f"val_loss={summary['final_val_loss']:.8f} "
            f"val_bpb={summary['final_val_bpb']:.8f}"
        )
    else:
        print("final_exact unavailable; run may still be incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
