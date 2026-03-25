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
    parser = argparse.ArgumentParser(
        description="Patch PR758 train_gpt.py with env-gated torch.compile support."
    )
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
        '    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)\n',
        '    compiled_logits = (\n'
        '        torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)\n'
        '        if args.compile_enabled else base_model.forward_logits\n'
        '    )\n',
    )

    text = replace_once(
        text,
        '    args = Hyperparameters()\n'
        '    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)\n',
        '    args = Hyperparameters()\n'
        '    if args.compile_enabled:\n'
        '        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)\n',
    )

    text = replace_once(
        text,
        '    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)\n'
        '    _needs_find_unused = args.value_residual or args.gated_attention\n'
        '    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=_needs_find_unused) if distributed else compiled_model\n',
        '    compiled_model = (\n'
        '        torch.compile(base_model, dynamic=False, fullgraph=True)\n'
        '        if args.compile_enabled else base_model\n'
        '    )\n'
        '    _needs_find_unused = args.value_residual or args.gated_attention\n'
        '    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=_needs_find_unused) if distributed else compiled_model\n',
    )

    text = replace_once(
        text,
        '    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)\n',
        '    compiled_eval = (\n'
        '        torch.compile(eval_model, dynamic=False, fullgraph=True)\n'
        '        if args.compile_enabled else eval_model\n'
        '    )\n',
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
