#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch


FORWARD_SHAPES = {
    "Q_proj": (98304, 512, 512),
    "K_proj": (98304, 512, 256),
    "V_proj": (98304, 512, 256),
    "Out_proj": (98304, 512, 512),
    "MLP_up": (98304, 512, 1536),
    "MLP_down": (98304, 1536, 512),
    "LM_head": (98304, 512, 1024),
    "Bigram": (98304, 128, 512),
}


def bench_mm(m: int, k: int, n: int, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    flops = 2 * m * k * n
    tflops = flops / ms / 1e9
    return ms, tflops


def bench_bmm(batch: int, m: int, k: int, n: int, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.bmm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.bmm(a, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    flops = 2 * batch * m * k * n
    tflops = flops / ms / 1e9
    return ms, tflops


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Parameter Golf GEMM shapes on the local GPU.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batched", action="store_true", help="Also benchmark batched Muon Newton-Schulz GEMMs.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"{'name':<10} {'ms':>10} {'TFLOPS':>10}")
    for name, (m, k, n) in FORWARD_SHAPES.items():
        ms, tflops = bench_mm(m, k, n, warmup=args.warmup, iters=args.iters)
        print(f"{name:<10} {ms:>10.4f} {tflops:>10.1f}")

    if not args.batched:
        return

    print("--- batched NS ---")
    for name, batch, m, k, n in [
        ("512x512", 22, 512, 512, 512),
        ("256x256", 22, 256, 256, 256),
        ("512x1536", 22, 512, 512, 1536),
    ]:
        bmm_ms, bmm_tflops = bench_bmm(batch, m, k, n, warmup=args.warmup, iters=args.iters)
        loop_ms = 0.0
        for _ in range(batch):
            ms, _ = bench_mm(m, k, n, warmup=2, iters=max(5, args.iters // 4))
            loop_ms += ms
        print(
            f"{name:<10} bmm_ms={bmm_ms:.4f} bmm_tf={bmm_tflops:.1f} "
            f"loop_ms={loop_ms:.4f} speedup={loop_ms / bmm_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
