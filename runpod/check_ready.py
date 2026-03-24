#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "runpod"


def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def extract_configs_from_wrapper(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"configs/runpod/[A-Za-z0-9_./-]+\.env", text)


def main() -> int:
    errors: list[str] = []
    infos: list[str] = []

    config_paths = sorted(CONFIG_DIR.glob("*.env"))
    if not config_paths:
        errors.append("no runpod config files found")
    configs = {path.name: parse_env(path) for path in config_paths}

    run_name_prefixes: dict[str, str] = {}
    for name, env in configs.items():
        prefix = env.get("RUN_NAME_PREFIX")
        if not prefix:
            errors.append(f"{name}: missing RUN_NAME_PREFIX")
            continue
        prev = run_name_prefixes.get(prefix)
        if prev is not None and prev != name:
            errors.append(f"duplicate RUN_NAME_PREFIX {prefix!r} in {prev} and {name}")
        run_name_prefixes[prefix] = name

    wrappers = [
        ROOT / "runpod" / "local_recover_export_chain.sh",
        ROOT / "runpod" / "local_recover_export_chain_8gpu.sh",
    ]
    for wrapper in wrappers:
        refs = extract_configs_from_wrapper(wrapper)
        if len(refs) < 5:
            errors.append(f"{wrapper.name}: expected at least 5 config references, found {len(refs)}")
        for ref in refs:
            path = ROOT / ref
            if not path.exists():
                errors.append(f"{wrapper.name}: missing referenced config {ref}")
        infos.append(f"{wrapper.name}: {len(refs)} config refs")

    for ladder_name in [
        "non_ttt_vrl_gptq_1gpu_export_prune17.env",
        "non_ttt_vrl_gptq_1gpu_export_prune20.env",
        "non_ttt_vrl_gptq_1gpu_export_prune23.env",
        "non_ttt_vrl_gptq_1gpu_export_prune26.env",
        "non_ttt_vrl_gptq_8gpu_export_prune17.env",
        "non_ttt_vrl_gptq_8gpu_export_prune20.env",
        "non_ttt_vrl_gptq_8gpu_export_prune23.env",
        "non_ttt_vrl_gptq_8gpu_export_prune26.env",
    ]:
        env = configs.get(ladder_name)
        if env is None:
            errors.append(f"missing ladder config {ladder_name}")
            continue
        if env.get("TRAIN_SCRIPT") != "candidates/non_ttt_vrl_gptq/train_gpt.py":
            errors.append(f"{ladder_name}: unexpected TRAIN_SCRIPT {env.get('TRAIN_SCRIPT')!r}")
        if "PRUNE_PCT" not in env:
            errors.append(f"{ladder_name}: missing PRUNE_PCT")

    train_cfg = configs.get("non_ttt_vrl_gptq_1gpu_long_prune14.env")
    if train_cfg is None:
        errors.append("missing non_ttt_vrl_gptq_1gpu_long_prune14.env")
    else:
        if train_cfg.get("SAVE_PRE_EXPORT_CHECKPOINT") != "1":
            errors.append("non_ttt_vrl_gptq_1gpu_long_prune14.env: SAVE_PRE_EXPORT_CHECKPOINT must be 1")

    train_cfg_8 = configs.get("non_ttt_vrl_gptq_8gpu_prune14.env")
    if train_cfg_8 is None:
        errors.append("missing non_ttt_vrl_gptq_8gpu_prune14.env")
    else:
        if train_cfg_8.get("SAVE_PRE_EXPORT_CHECKPOINT") != "1":
            errors.append("non_ttt_vrl_gptq_8gpu_prune14.env: SAVE_PRE_EXPORT_CHECKPOINT must be 1")

    if errors:
        print("runpod readiness check: FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("runpod readiness check: OK")
    for info in infos:
        print(f"- {info}")
    print(f"- verified {len(configs)} config files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
