#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import lzma
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F


def load_module(train_gpt_path: Path):
    spec = importlib.util.spec_from_file_location("record_train_gpt", train_gpt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {train_gpt_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["record_train_gpt"] = module
    spec.loader.exec_module(module)
    return module


def build_eval_model(mod, args, device: torch.device, deq_state: dict[str, torch.Tensor]) -> torch.nn.Module:
    eval_model = mod.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for module in eval_model.modules():
        if isinstance(module, mod.CastedLinear):
            module.float()
    mod.restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


class OnlineNgramCache:
    def __init__(self, vocab_size: int, max_n: int = 5) -> None:
        self.vocab_size = vocab_size
        self.max_n = max_n
        self.counts: list[dict[tuple[int, ...], dict[int, int]]] = [{} for _ in range(max_n + 1)]

    def update_from_list(self, token_ids: list[int]) -> None:
        for i, token in enumerate(token_ids):
            for n in range(2, min(self.max_n + 1, i + 2)):
                ctx = tuple(token_ids[i - n + 1 : i])
                d = self.counts[n].get(ctx)
                if d is None:
                    d = {}
                    self.counts[n][ctx] = d
                d[token] = d.get(token, 0) + 1

    def logprob_target(self, context: list[int], target: int, min_count: int = 3) -> float | None:
        for n in range(min(self.max_n, len(context) + 1), 1, -1):
            ctx = tuple(context[-(n - 1) :]) if n > 1 else ()
            d = self.counts[n].get(ctx)
            if d is None:
                continue
            total = sum(d.values())
            if total < min_count:
                continue
            cnt = d.get(target, 0)
            if cnt > 0:
                return math.log(cnt / total)
        return None


def eval_val_ngram(
    mod,
    args,
    base_model: torch.nn.Module,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    *,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    ngram_lambda: float = 0.15,
    ngram_max_n: int = 5,
    confidence_threshold: float = 0.5,
    min_count: int = 3,
    ngram_adapt_enabled: bool = False,
    ngram_adapt_lr: float = 0.0003,
    ngram_adapt_decay: float = 0.001,
    max_windows: int = 0,
    log=print,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    log_conf_thresh = math.log(confidence_threshold) if confidence_threshold < 1.0 else 0.0
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    if max_windows > 0:
        window_starts = window_starts[:max_windows]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ngram = OnlineNgramCache(args.vocab_size, max_n=ngram_max_n)
    ngram_improvements = 0
    ngram_attempts = 0
    ngram_skipped = 0
    lam = ngram_lambda
    log_1_minus_lam = math.log(1.0 - lam)
    log_lam = math.log(lam)
    ngram_adapt_optimizer = None
    global_weights = None

    if ngram_adapt_enabled:
        base_model.train()
        num_layers = len(base_model.blocks)
        update_params = []
        target_layers = list(range(max(0, num_layers - 3), num_layers))
        for idx in target_layers:
            for p in base_model.blocks[idx].parameters():
                if p.requires_grad:
                    update_params.append(p)
        global_weights = {id(p): p.data.clone() for p in update_params}
        ngram_adapt_optimizer = torch.optim.RMSprop(update_params, lr=ngram_adapt_lr, alpha=0.99, eps=1e-8)
    else:
        base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    log(
        "ngram_eval:start "
        f"stride={stride} lambda={ngram_lambda} max_n={ngram_max_n} "
        f"confidence_threshold={confidence_threshold} min_count={min_count} "
        f"adapt={int(ngram_adapt_enabled)}"
    )
    for bi in range(0, len(window_starts), batch_seqs):
        batch_ws = window_starts[bi : bi + batch_seqs]
        bsz = len(batch_ws)
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
        logits_f = logits.float()
        nll = F.cross_entropy(
            logits_f.reshape(-1, logits_f.size(-1)),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)

        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s = 0 if ws == 0 else max(wlen - stride, 0)
            scored_nll = nll[i, s:wlen].to(torch.float64).clone()
            if ws > 0:
                log_sm = F.log_softmax(logits_f[i, s:wlen], dim=-1)
                max_logp_cpu = log_sm.max(dim=-1).values.cpu().tolist()
                x_cpu = x_batch[i, :wlen].cpu().tolist()
                y_cpu = y_batch[i, s:wlen].cpu().tolist()
                uncertain_indices: list[int] = []
                prev_contexts: list[list[int]] = []
                targets: list[int] = []
                n_scored = wlen - s
                for t_off in range(n_scored):
                    if max_logp_cpu[t_off] > log_conf_thresh:
                        ngram_skipped += 1
                        continue
                    t_idx = s + t_off
                    uncertain_indices.append(t_off)
                    ctx_start = max(0, t_idx - ngram_max_n + 1)
                    prev_contexts.append(x_cpu[ctx_start : t_idx + 1])
                    targets.append(y_cpu[t_off])

                if uncertain_indices:
                    ng_logps = [ngram.logprob_target(ctx, tgt, min_count=min_count) for ctx, tgt in zip(prev_contexts, targets)]
                    unc_idx_t = torch.tensor(uncertain_indices, dtype=torch.long, device=log_sm.device)
                    tgt_t = torch.tensor(targets, dtype=torch.long, device=log_sm.device)
                    model_logps = log_sm[unc_idx_t, tgt_t].cpu().tolist()
                    for j, t_off in enumerate(uncertain_indices):
                        ng_lp = ng_logps[j]
                        if ng_lp is None:
                            continue
                        ngram_attempts += 1
                        model_lp = model_logps[j]
                        a = log_1_minus_lam + model_lp
                        b = log_lam + ng_lp
                        mixed_lp = max(a, b) + math.log1p(math.exp(-abs(a - b)))
                        new_nll = -mixed_lp
                        old_nll = scored_nll[t_off].item()
                        if new_nll < old_nll:
                            scored_nll[t_off] = new_nll
                            ngram_improvements += 1

            loss_sum += scored_nll.sum()
            token_count += float(wlen - s)
            tgt = y_batch[i, s:wlen]
            prev = x_batch[i, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum()

        for i in range(bsz):
            wlen = wlens[i]
            toks = x_batch[i, :wlen].cpu().tolist()
            toks.append(y_batch[i, wlen - 1].item())
            ngram.update_from_list(toks)

        if ngram_adapt_enabled and ngram_adapt_optimizer is not None:
            last_wlen = wlens[-1]
            last_s = 0 if batch_ws[-1] == 0 else max(last_wlen - stride, 0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                adapt_logits = compiled_logits(x_batch[-1:, :last_wlen])
            adapt_loss = F.cross_entropy(
                adapt_logits[0, last_s:last_wlen].float(),
                y_batch[-1, last_s:last_wlen],
            )
            ngram_adapt_optimizer.zero_grad()
            adapt_loss.backward()
            ngram_adapt_optimizer.step()
            if global_weights is not None:
                with torch.no_grad():
                    for p in ngram_adapt_optimizer.param_groups[0]["params"]:
                        pid = id(p)
                        if pid in global_weights:
                            p.data.mul_(1.0 - ngram_adapt_decay).add_(global_weights[pid], alpha=ngram_adapt_decay)

        if bi % (batch_seqs * 50) == 0 and token_count.item() > 0:
            rl = loss_sum.item() / token_count.item()
            rb = (rl / math.log(2.0)) * (token_count.item() / max(byte_count.item(), 1))
            pct = 100.0 * bi / max(len(window_starts), 1)
            hit = ngram_improvements / max(ngram_attempts, 1) * 100
            skip = ngram_skipped / max(ngram_skipped + ngram_attempts + 1, 1) * 100
            suffix = " +ngram_adapt" if ngram_adapt_enabled else ""
            log(f"  ngram [{pct:5.1f}%] bpb={rb:.6f} hit={hit:.1f}% skip={skip:.0f}%{suffix}")

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    log(f"  ngram: {ngram_improvements}/{ngram_attempts} improved, {ngram_skipped} skipped")
    return val_loss, bits_per_token * tokens_per_byte


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved artifact with backward-looking n-gram cache mixing.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/home/aryang9/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--data-path", type=Path, default=Path("/home/aryang9/parameter-golf/data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--template-path", type=Path)
    parser.add_argument("--train-gpt-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--bigram-vocab-size", type=int, default=1536)
    parser.add_argument("--value-residual", type=int, choices=[0, 1], default=None)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--ngram-lambda", type=float, default=0.15)
    parser.add_argument("--ngram-max-n", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--ngram-adapt-enabled", action="store_true")
    parser.add_argument("--ngram-adapt-lr", type=float, default=0.0003)
    parser.add_argument("--ngram-adapt-decay", type=float, default=0.001)
    parser.add_argument("--max-windows", type=int, default=0)
    args_ns = parser.parse_args()

    run_dir = args_ns.run_dir.resolve()
    artifact_path = (args_ns.artifact_path or run_dir / "final_model.int6.ptz").resolve()
    template_path = (args_ns.template_path or run_dir / "final_model.pt").resolve()
    train_gpt_path = (args_ns.train_gpt_path or run_dir / "train_gpt.py").resolve()
    log_path = args_ns.log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mod = load_module(train_gpt_path)
    args = mod.Hyperparameters()
    args.data_path = str(args_ns.data_path.resolve())
    args.train_files = os.path.join(args.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")
    args.tokenizer_path = str(args_ns.tokenizer_path.resolve())
    args.bigram_vocab_size = args_ns.bigram_vocab_size
    if args_ns.value_residual is not None:
        args.value_residual = bool(args_ns.value_residual)

    device = torch.device(args_ns.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            print(msg, file=f)

    log("resume_ngram_eval:start")
    log(f"resume_ngram_eval:artifact_path={artifact_path}")
    log(f"resume_ngram_eval:template_path={template_path}")
    log(f"resume_ngram_eval:train_gpt_path={train_gpt_path}")
    log(f"resume_ngram_eval:device={device}")
    log(f"resume_ngram_eval:torch={torch.__version__}")
    log(
        json.dumps(
            {
                "stride": args_ns.stride,
                "ngram_lambda": args_ns.ngram_lambda,
                "ngram_max_n": args_ns.ngram_max_n,
                "confidence_threshold": args_ns.confidence_threshold,
                "min_count": args_ns.min_count,
                "ngram_adapt_enabled": int(args_ns.ngram_adapt_enabled),
                "ngram_adapt_lr": args_ns.ngram_adapt_lr,
                "ngram_adapt_decay": args_ns.ngram_adapt_decay,
                "batch_seqs": args_ns.batch_seqs,
                "bigram_vocab_size": args.bigram_vocab_size,
                "value_residual": int(bool(args.value_residual)),
            },
            sort_keys=True,
        )
    )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = mod.load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(sp, args.vocab_size, device)

    export_sd = torch.load(template_path, map_location="cpu")
    with open(artifact_path, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    unbanked_sd = mod._unbank_state_dict(export_sd, args.num_layers)
    deq_unbanked = mod.dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = mod._rebank_state_dict(deq_unbanked, args.num_layers, export_sd)
    eval_model = build_eval_model(mod, args, device, deq_state)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val_ngram(
        mod,
        args,
        eval_model,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        stride=args_ns.stride,
        batch_seqs=args_ns.batch_seqs,
        eval_seq_len=effective_eval_seq_len,
        ngram_lambda=args_ns.ngram_lambda,
        ngram_max_n=args_ns.ngram_max_n,
        confidence_threshold=args_ns.confidence_threshold,
        min_count=args_ns.min_count,
        ngram_adapt_enabled=args_ns.ngram_adapt_enabled,
        ngram_adapt_lr=args_ns.ngram_adapt_lr,
        ngram_adapt_decay=args_ns.ngram_adapt_decay,
        max_windows=args_ns.max_windows,
        log=log,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"final_ngram_eval val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} stride:{args_ns.stride} eval_time:{elapsed_ms:.0f}ms")
    log(f"final_ngram_eval_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    log(
        f"resume_ngram_eval:peak_memory_allocated_mib={torch.cuda.max_memory_allocated() // 1024 // 1024} "
        f"reserved_mib={torch.cuda.max_memory_reserved() // 1024 // 1024}"
    )


if __name__ == "__main__":
    main()
