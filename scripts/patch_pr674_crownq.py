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
    parser = argparse.ArgumentParser(description="Patch PR674 train_gpt.py with PR692-style CROWN-Q training penalty.")
    parser.add_argument("train_gpt", type=Path)
    args = parser.parse_args()

    path = args.train_gpt
    text = path.read_text()

    text = replace_once(
        text,
        '    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))\n'
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.5))\n'
        '    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))\n',
        '    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))\n'
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.5))\n'
        '    crownq_lambda = float(os.environ.get("CROWNQ_LAMBDA", 0.01))\n'
        '    crownq_warmdown_only = bool(int(os.environ.get("CROWNQ_WARMDOWN_ONLY", "1")))\n'
        '    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))\n',
    )

    text = replace_once(
        text,
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            loss.backward()\n",
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "                loss = model(x, y)\n"
        "            if CastedLinear._qat_enabled and args.crownq_lambda > 0 and (not args.crownq_warmdown_only or scale < 1.0):\n"
        "                crownq_penalty = torch.zeros((), device=device)\n"
        "                for m in base_model.modules():\n"
        "                    if isinstance(m, CastedLinear) and m.weight.ndim == 2:\n"
        "                        w = m.weight.float()\n"
        "                        row_clip = torch.quantile(w.abs(), 0.9995, dim=1).clamp(min=1e-10)\n"
        "                        delta = row_clip / 31.0\n"
        "                        quant_var = (delta ** 2) / 12.0\n"
        "                        h_proxy = (w ** 2).mean(dim=1)\n"
        "                        crownq_penalty = crownq_penalty + (h_proxy * quant_var).sum()\n"
        "                loss = loss + args.crownq_lambda * crownq_penalty\n"
        "            train_loss += loss.detach()\n"
        "            loss.backward()\n",
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
