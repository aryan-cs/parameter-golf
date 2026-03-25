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


def eval_val_sliding_ttt_ngram(
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
    ngram_lambda: float = 0.15,
    ngram_max_n: int = 5,
    confidence_threshold: float = 0.5,
    min_count: int = 3,
    max_chunks: int = 0,
    log=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    log_conf_thresh = math.log(confidence_threshold) if confidence_threshold < 1.0 else 0.0
    lam = ngram_lambda
    log_1_minus_lam = math.log(1.0 - lam)
    log_lam = math.log(lam)

    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    if max_chunks > 0:
        chunk_windows = chunk_windows[:max_chunks]
        num_chunks = len(chunk_windows)

    log(
        "ttt_ngram:start "
        f"chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} "
        f"stride={stride} ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
        f"freeze_blocks={args.ttt_freeze_blocks} ngram_lambda={ngram_lambda} "
        f"ngram_max_n={ngram_max_n} confidence_threshold={confidence_threshold}"
    )

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ngram = OnlineNgramCache(args.vocab_size, max_n=ngram_max_n)
    ngram_improvements = 0
    ngram_attempts = 0
    ngram_skipped = 0

    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(windows), batch_seqs):
                batch_ws = windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
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

        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, chunk_seqs)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if ci % 10 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            hit = ngram_improvements / max(ngram_attempts, 1) * 100
            skip = ngram_skipped / max(ngram_skipped + ngram_attempts + 1, 1) * 100
            log(f"  ttt_ngram_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} hit={hit:.1f}% skip={skip:.0f}% time={elapsed:.1f}s")

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log(
        f"ttt_ngram:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
        f"elapsed={time.perf_counter() - t0:.1f}s improved={ngram_improvements}/{ngram_attempts} skipped={ngram_skipped}"
    )
    return val_loss, val_bpb


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved artifact with backward-looking n-gram cache + legal TTT.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/home/aryang9/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--data-path", type=Path, default=Path("/home/aryang9/parameter-golf/data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--template-path", type=Path)
    parser.add_argument("--train-gpt-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bigram-vocab-size", type=int, default=1536)
    parser.add_argument("--value-residual", type=int, choices=[0, 1], default=None)
    parser.add_argument("--ttt-freeze-blocks", type=int, default=0)
    parser.add_argument("--ttt-lr", type=float, default=0.0025)
    parser.add_argument("--ttt-epochs", type=int, default=3)
    parser.add_argument("--ttt-chunk-tokens", type=int, default=32768)
    parser.add_argument("--ttt-momentum", type=float, default=0.9)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--ngram-lambda", type=float, default=0.15)
    parser.add_argument("--ngram-max-n", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=0)
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
    args.ttt_enabled = True
    args.ttt_freeze_blocks = args_ns.ttt_freeze_blocks
    args.ttt_lr = args_ns.ttt_lr
    args.ttt_epochs = args_ns.ttt_epochs
    args.ttt_chunk_tokens = args_ns.ttt_chunk_tokens
    args.ttt_momentum = args_ns.ttt_momentum
    args.ttt_grad_clip = args_ns.ttt_grad_clip
    args.ttt_batch_seqs = args_ns.batch_seqs

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

    log("resume_ttt_ngram_eval:start")
    log(f"resume_ttt_ngram_eval:artifact_path={artifact_path}")
    log(f"resume_ttt_ngram_eval:template_path={template_path}")
    log(f"resume_ttt_ngram_eval:train_gpt_path={train_gpt_path}")
    log(f"resume_ttt_ngram_eval:device={device}")
    log(f"resume_ttt_ngram_eval:torch={torch.__version__}")
    log(
        json.dumps(
            {
                "bigram_vocab_size": args.bigram_vocab_size,
                "value_residual": int(bool(args.value_residual)),
                "ttt_lr": args.ttt_lr,
                "ttt_epochs": args.ttt_epochs,
                "ttt_chunk_tokens": args.ttt_chunk_tokens,
                "ttt_freeze_blocks": args.ttt_freeze_blocks,
                "ttt_momentum": args.ttt_momentum,
                "ttt_batch_seqs": args.ttt_batch_seqs,
                "ttt_grad_clip": args.ttt_grad_clip,
                "stride": args_ns.stride,
                "ngram_lambda": args_ns.ngram_lambda,
                "ngram_max_n": args_ns.ngram_max_n,
                "confidence_threshold": args_ns.confidence_threshold,
                "min_count": args_ns.min_count,
                "max_chunks": args_ns.max_chunks,
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
    val_loss, val_bpb = eval_val_sliding_ttt_ngram(
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
        ngram_lambda=args_ns.ngram_lambda,
        ngram_max_n=args_ns.ngram_max_n,
        confidence_threshold=args_ns.confidence_threshold,
        min_count=args_ns.min_count,
        max_chunks=args_ns.max_chunks,
        log=log,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"legal_ttt_ngram val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} eval_time:{elapsed_ms:.0f}ms")
    log(f"legal_ttt_ngram_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    log(
        f"resume_ttt_ngram_eval:peak_memory_allocated_mib={torch.cuda.max_memory_allocated() // 1024 // 1024} "
        f"reserved_mib={torch.cuda.max_memory_reserved() // 1024 // 1024}"
    )


if __name__ == "__main__":
    main()
