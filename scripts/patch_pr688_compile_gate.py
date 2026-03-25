#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match for snippet, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch PR688 train_gpt.py with an env-gated torch.compile path.")
    parser.add_argument("train_gpt", type=Path)
    args = parser.parse_args()

    path = args.train_gpt
    text = path.read_text()

    text = replace_once(
        text,
        '    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))\n'
        '    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))\n',
        '    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))\n'
        '    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))\n'
        '    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))\n',
    )

    text = replace_once(
        text,
        '    base_model.eval()\n'
        '    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)\n'
        '\n'
        '    # Pre-compile: dummy forward+backward with TTT shapes to warm the compile cache\n'
        '    if rank == 0:\n'
        '        print("  ttt: pre-compiling forward+backward kernels...", flush=True)\n'
        '    _dummy_x = torch.zeros(1, seq_len, dtype=torch.int64, device=device)\n'
        '    _dummy_y = torch.zeros(1, seq_len, dtype=torch.int64, device=device)\n'
        '    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
        '        _dummy_logits = base_model.forward_logits(_dummy_x)\n'
        '        _dummy_loss = F.cross_entropy(_dummy_logits.reshape(-1, _dummy_logits.size(-1)), _dummy_y.reshape(-1))\n'
        '    _dummy_loss.backward()\n'
        '    base_model.zero_grad(set_to_none=True)\n'
        '    if rank == 0:\n'
        '        print("  ttt: pre-compile done", flush=True)\n',
        '    base_model.eval()\n'
        '    if args.compile_enabled:\n'
        '        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)\n'
        '\n'
        '        # Pre-compile: dummy forward+backward with TTT shapes to warm the compile cache\n'
        '        if rank == 0:\n'
        '            print("  ttt: pre-compiling forward+backward kernels...", flush=True)\n'
        '        _dummy_x = torch.zeros(1, seq_len, dtype=torch.int64, device=device)\n'
        '        _dummy_y = torch.zeros(1, seq_len, dtype=torch.int64, device=device)\n'
        '        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n'
        '            _dummy_logits = base_model.forward_logits(_dummy_x)\n'
        '            _dummy_loss = F.cross_entropy(_dummy_logits.reshape(-1, _dummy_logits.size(-1)), _dummy_y.reshape(-1))\n'
        '        _dummy_loss.backward()\n'
        '        base_model.zero_grad(set_to_none=True)\n'
        '        if rank == 0:\n'
        '            print("  ttt: pre-compile done", flush=True)\n'
        '    else:\n'
        '        compiled_logits = base_model.forward_logits\n'
        '        if rank == 0:\n'
        '            print("  ttt: compile disabled", flush=True)\n',
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
