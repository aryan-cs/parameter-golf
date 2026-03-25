#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import lzma
from pathlib import Path

import torch
from torch import Tensor


def load_module(train_gpt_path: Path):
    spec = importlib.util.spec_from_file_location("artifact_train_gpt", train_gpt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {train_gpt_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_pcts(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one clip percentile is required")
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"clip percentile must be in (0, 1], got {v}")
    return vals


def quantize_int6_per_row_grid(t: Tensor, clip_pcts: list[float], clip_range: int = 31) -> tuple[Tensor, Tensor, float, float]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err, best_pct = None, None, float("inf"), clip_pcts[0]
        for pct in clip_pcts:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err, best_pct = q, s, err, pct
        assert best_q is not None and best_s is not None
        return best_q, best_s, best_err, best_pct
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale, 0.0, 1.0


def check_reference_compatibility(reference_payload: dict[str, object], template_sd: dict[str, Tensor]) -> list[dict[str, object]]:
    w = reference_payload["w"]
    sentinel_checks = [
        ("skip_weights", "skip_weights"),
        ("tok_emb.weight.q", "tok_emb.weight"),
        ("bigram.embed.weight.q", "bigram.embed.weight"),
        ("bigram.proj.weight.q", "bigram.proj.weight"),
    ]
    mismatches: list[dict[str, object]] = []
    for artifact_key, template_key in sentinel_checks:
        if artifact_key not in w or template_key not in template_sd:
            continue
        artifact_shape = tuple(w[artifact_key].shape)
        template_shape = tuple(template_sd[template_key].shape)
        if artifact_shape != template_shape:
            mismatches.append(
                {
                    "artifact_key": artifact_key,
                    "artifact_shape": artifact_shape,
                    "template_key": template_key,
                    "template_shape": template_shape,
                }
            )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-quantize a saved FP checkpoint with a custom GPTQ-lite int6 clip grid.")
    parser.add_argument("--train-gpt-path", type=Path, required=True)
    parser.add_argument("--template-path", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--int6-clip-pcts", type=str, default="0.9990,0.9995,0.9999,0.99999,1.0")
    parser.add_argument("--int6-cats", type=str, default="mlp,attn")
    parser.add_argument("--lzma-preset", type=int, default=6)
    parser.add_argument("--verify-roundtrip", action="store_true")
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    args = parser.parse_args()

    clip_pcts = parse_pcts(args.int6_clip_pcts)
    int6_cats = {x.strip() for x in args.int6_cats.split(",") if x.strip()}

    mod = load_module(args.train_gpt_path)
    sd_cpu = torch.load(args.template_path, map_location="cpu")
    if not isinstance(sd_cpu, dict):
        raise TypeError(f"expected state dict at {args.template_path}, got {type(sd_cpu)}")

    reference_mismatches: list[dict[str, object]] = []
    if args.reference_artifact and args.reference_artifact.exists():
        reference_payload = torch.load(
            io.BytesIO(lzma.decompress(args.reference_artifact.read_bytes())),
            map_location="cpu",
        )
        reference_mismatches = check_reference_compatibility(reference_payload, sd_cpu)
        if reference_mismatches and not args.allow_reference_mismatch:
            raise ValueError(
                "reference artifact does not match template checkpoint: "
                + json.dumps(reference_mismatches, sort_keys=True)
            )

    num_layers = int(sd_cpu["qo_bank"].shape[0] // 2) if "qo_bank" in sd_cpu else int(mod.Hyperparameters().num_layers)
    unbanked_sd = mod._unbank_state_dict(sd_cpu, num_layers)
    control_patterns = tuple(getattr(mod, "CONTROL_TENSOR_NAME_PATTERNS", ()))

    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    best_pct_counts: dict[str, int] = {}
    int6_err_sum = 0.0
    int6_err_count = 0

    for name, tensor in unbanked_sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = mod._classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in control_patterns):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s, err, best_pct = quantize_int6_per_row_grid(t, clip_pcts)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
            best_pct_counts[f"{best_pct:.6f}"] = best_pct_counts.get(f"{best_pct:.6f}", 0) + 1
            int6_err_sum += err
            int6_err_count += 1
        else:
            q, s = mod.quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}

    quant_buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=args.lzma_preset)

    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.output_artifact.write_bytes(quant_blob)

    summary = {
        "template_path": str(args.template_path.resolve()),
        "output_artifact": str(args.output_artifact.resolve()),
        "artifact_bytes": len(quant_blob),
        "lzma_preset": args.lzma_preset,
        "int6_clip_pcts": clip_pcts,
        "int6_cats": sorted(int6_cats),
        "int6_mean_mse": (int6_err_sum / int6_err_count) if int6_err_count else 0.0,
        "best_pct_counts": best_pct_counts,
    }
    if args.reference_artifact and args.reference_artifact.exists():
        ref_bytes = args.reference_artifact.stat().st_size
        summary["reference_artifact"] = str(args.reference_artifact.resolve())
        summary["reference_bytes"] = ref_bytes
        summary["bytes_delta_vs_reference"] = len(quant_blob) - ref_bytes
        summary["reference_mismatches"] = reference_mismatches

    if args.verify_roundtrip:
        quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
        deq_unbanked = mod.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
        _ = mod._rebank_state_dict(deq_unbanked, num_layers, sd_cpu)
        summary["verify_roundtrip"] = True

    print("requantize_artifact:start")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
