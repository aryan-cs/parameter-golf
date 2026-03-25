#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match for snippet, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch PR685 train_gpt.py to use legal mean-prob multipass aggregation.")
    parser.add_argument("train_gpt", type=Path)
    parser.add_argument("--power", type=float, default=1.0, help="Fixed probability power mean; 1.0 = arithmetic mean of probabilities.")
    args = parser.parse_args()
    if not math.isfinite(args.power) or args.power <= 0.0:
        raise SystemExit("--power must be a finite positive float")

    path = args.train_gpt
    text = path.read_text()
    power = args.power
    if abs(power - 1.0) < 1e-9:
        tag = "meanprob"
    else:
        tag = f"powmean{('%g' % power).replace('.', 'p')}"

    text = replace_once(
        text,
        '    base_sd = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}\n'
        "    best_nll = torch.full((total_val,), float('inf'), device=device, dtype=torch.float32)\n"
        "    t_mp = time.perf_counter()\n"
        '    log0(f"phase2_multipass: {ttt_passes} passes, lr={ttt_lr}")\n',
        '    base_sd = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}\n'
        f"    agg_power = {power:.8g}\n"
        "    prob_sum = torch.zeros((total_val,), device=device, dtype=torch.float32)\n"
        "    pass_count = torch.zeros((total_val,), device=device, dtype=torch.uint8)\n"
        "    t_mp = time.perf_counter()\n"
        f'    log0(f"phase2_multipass_{tag}: {{ttt_passes}} passes, lr={{ttt_lr}}, power={{agg_power}}")\n',
    )

    text = replace_once(
        text,
        "            if vm.any(): best_nll[indices[vm]] = torch.minimum(best_nll[indices[vm]], nll[vm])\n",
        "            if vm.any():\n"
        "                idx = indices[vm]\n"
        "                prob_sum[idx] += torch.exp(-agg_power * nll[vm].to(torch.float32))\n"
        "                pass_count[idx] += 1\n",
    )

    text = replace_once(
        text,
        "        if master_process:\n"
        "            scored = (best_nll<float('inf')).sum().item()\n"
        "            avg = best_nll[best_nll<float('inf')].mean().item() if scored>0 else 0\n"
        '            log0(f"  pass {pass_idx}: scored={scored} avg_nll={avg:.4f}")\n'
        "    my_mask = torch.zeros(total_val, device=device, dtype=torch.bool)\n"
        "    my_mask[rank_start:min(rank_end,total_val)] = True\n"
        "    my_valid = my_mask & (best_nll<float('inf'))\n"
        "    loss_sum = best_nll[my_valid].to(torch.float64).sum()\n",
        "        if master_process:\n"
        "            scored = (pass_count > 0).sum().item()\n"
        "            if scored > 0:\n"
        "                agg_prob = (prob_sum[pass_count > 0] / pass_count[pass_count > 0].to(torch.float32)).clamp_min(1e-12)\n"
        "                agg_nll = -(1.0 / agg_power) * torch.log(agg_prob)\n"
        "                avg = agg_nll.mean().item()\n"
        "            else:\n"
        "                avg = 0.0\n"
        f'            log0(f"  pass {{pass_idx}}: scored={{scored}} avg_{tag}_nll={{avg:.4f}}")\n'
        "    my_mask = torch.zeros(total_val, device=device, dtype=torch.bool)\n"
        "    my_mask[rank_start:min(rank_end,total_val)] = True\n"
        "    my_valid = my_mask & (pass_count > 0)\n"
        "    agg_prob = (prob_sum[my_valid].to(torch.float64) / pass_count[my_valid].to(torch.float64)).clamp_min(1e-12)\n"
        "    loss_sum = (-(1.0 / agg_power) * torch.log(agg_prob)).sum()\n",
    )

    text = replace_once(
        text,
        '    log0(f"phase2_multipass: done in {1000*(time.perf_counter()-t_mp):.0f}ms")\n'
        '    log0(f"chained_ttt val_loss:{final_loss:.4f} val_bpb:{final_bpb:.4f}")\n',
        f'    log0(f"phase2_multipass_{tag}: done in {{1000*(time.perf_counter()-t_mp):.0f}}ms")\n'
        f'    log0(f"chained_ttt_{tag} val_loss:{{final_loss:.4f}} val_bpb:{{final_bpb:.4f}}")\n',
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
