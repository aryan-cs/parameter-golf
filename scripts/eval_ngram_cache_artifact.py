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
        use_swiglu=getattr(args, "use_swiglu", False),
        swiglu_half_dim=getattr(args, "swiglu_half_dim", 1024),
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

    def lookup_target(self, context: list[int], target: int, min_count: int = 3) -> tuple[float, int, int] | None:
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
                return math.log(cnt / total), n, total
        return None

    def logprob_target(self, context: list[int], target: int, min_count: int = 3) -> float | None:
        lookup = self.lookup_target(context, target, min_count=min_count)
        return None if lookup is None else lookup[0]


class PackedOnlineNgramCache:
    def __init__(self, vocab_size: int, max_n: int = 5) -> None:
        self.vocab_size = vocab_size
        self.max_n = max_n
        self.token_bits = max(1, int(vocab_size - 1).bit_length())
        self.use_bitpack = (1 << self.token_bits) == vocab_size
        self.totals: list[dict[int, int]] = [{} for _ in range(max_n + 1)]
        self.counts: list[dict[int, int]] = [{} for _ in range(max_n + 1)]

    def _append_token(self, code: int, token: int) -> int:
        if self.use_bitpack:
            return (code << self.token_bits) | int(token)
        return code * self.vocab_size + int(token)

    def _encode_context(self, token_ids: list[int], start: int, end: int) -> int:
        code = 0
        for i in range(start, end):
            code = self._append_token(code, token_ids[i])
        return code

    def update_from_list(self, token_ids: list[int]) -> None:
        for i, token in enumerate(token_ids):
            for n in range(2, min(self.max_n + 1, i + 2)):
                ctx_code = self._encode_context(token_ids, i - n + 1, i)
                totals_n = self.totals[n]
                counts_n = self.counts[n]
                totals_n[ctx_code] = totals_n.get(ctx_code, 0) + 1
                joint = self._append_token(ctx_code, token)
                counts_n[joint] = counts_n.get(joint, 0) + 1

    def lookup_target(self, context: list[int], target: int, min_count: int = 3) -> tuple[float, int, int] | None:
        for n in range(min(self.max_n, len(context) + 1), 1, -1):
            ctx_code = self._encode_context(context, len(context) - (n - 1), len(context))
            total = self.totals[n].get(ctx_code, 0)
            if total < min_count:
                continue
            cnt = self.counts[n].get(self._append_token(ctx_code, target), 0)
            if cnt > 0:
                return math.log(cnt / total), n, total
        return None

    def logprob_target(self, context: list[int], target: int, min_count: int = 3) -> float | None:
        lookup = self.lookup_target(context, target, min_count=min_count)
        return None if lookup is None else lookup[0]


def build_ngram_cache(vocab_size: int, max_n: int, *, packed: bool) -> OnlineNgramCache | PackedOnlineNgramCache:
    if packed:
        return PackedOnlineNgramCache(vocab_size, max_n=max_n)
    return OnlineNgramCache(vocab_size, max_n=max_n)


def build_hashed_ngram_keys(
    val_np: np.ndarray,
    token_positions: np.ndarray,
    *,
    order: int,
    buckets: int,
    primes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ctx_width = order - 1
    ctx_hash = np.zeros((len(token_positions),), dtype=np.uint64)
    for k in range(ctx_width):
        tok = val_np[token_positions - (ctx_width - 1 - k)].astype(np.uint64)
        ctx_hash ^= tok * primes[k % len(primes)]
    mask = np.uint64(buckets - 1)
    ctx_key = (ctx_hash & mask).astype(np.int64)
    tgt_np = val_np[token_positions + 1].astype(np.uint64)
    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
    return ctx_key, full_key


def parse_confidence_schedule(spec: str, default_threshold: float) -> list[tuple[float, float]]:
    schedule = [(0.0, default_threshold)]
    if not spec.strip():
        return schedule
    parsed: list[tuple[float, float]] = []
    for item in spec.split(","):
        frac_s, value_s = item.strip().split(":", 1)
        frac = min(max(float(frac_s), 0.0), 1.0)
        value = min(max(float(value_s), 0.0), 1.0)
        parsed.append((frac, value))
    parsed.sort()
    if parsed[0][0] > 0.0:
        parsed.insert(0, (0.0, default_threshold))
    return parsed


def confidence_for_progress(schedule: list[tuple[float, float]], progress: float) -> float:
    current = schedule[0][1]
    for frac, value in schedule:
        if progress + 1e-12 >= frac:
            current = value
        else:
            break
    return current


def confidence_to_log_threshold(confidence_threshold: float) -> float:
    if confidence_threshold <= 0.0:
        return float("-inf")
    if confidence_threshold >= 1.0:
        return 0.0
    return math.log(confidence_threshold)


def parse_lambda_schedule(spec: str, default_lambda: float) -> list[tuple[float, float]]:
    schedule = [(0.0, default_lambda)]
    if not spec.strip():
        return schedule
    parsed: list[tuple[float, float]] = []
    for item in spec.split(","):
        frac_s, value_s = item.strip().split(":", 1)
        frac = min(max(float(frac_s), 0.0), 1.0)
        value = float(value_s)
        if not (0.0 < value < 1.0):
            raise ValueError(f"lambda schedule values must lie in (0, 1), got {value}")
        parsed.append((frac, value))
    parsed.sort()
    if parsed[0][0] > 0.0:
        parsed.insert(0, (0.0, default_lambda))
    return parsed


def lambda_for_progress(schedule: list[tuple[float, float]], progress: float) -> float:
    current = schedule[0][1]
    for frac, value in schedule:
        if progress + 1e-12 >= frac:
            current = value
        else:
            break
    return current


def parse_order_lambdas(spec: str, max_n: int, default_lambda: float) -> dict[int, float]:
    order_lambdas = {n: default_lambda for n in range(2, max_n + 1)}
    if not spec.strip():
        return order_lambdas
    for item in spec.split(","):
        order_s, value_s = item.strip().split(":", 1)
        order = int(order_s)
        value = float(value_s)
        if order < 2 or order > max_n:
            continue
        if not (0.0 < value < 1.0):
            raise ValueError(f"order lambda must lie in (0, 1), got {value} for n={order}")
        order_lambdas[order] = value
    return order_lambdas


def format_confidence_schedule(schedule: list[tuple[float, float]]) -> str:
    return ",".join(f"{frac:.2f}:{value:.2f}" for frac, value in schedule)


def format_lambda_schedule(schedule: list[tuple[float, float]]) -> str:
    return ",".join(f"{frac:.2f}:{value:.3f}" for frac, value in schedule)


def format_order_lambdas(order_lambdas: dict[int, float]) -> str:
    return ",".join(f"{order}:{order_lambdas[order]:.3f}" for order in sorted(order_lambdas))


def gate_value_to_log_threshold(gate_threshold: float) -> float:
    return confidence_to_log_threshold(gate_threshold)


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
    gate_mode: str = "max",
    min_count: int = 3,
    apply_mode: str = "improve_only",
    ngram_adapt_enabled: bool = False,
    ngram_adapt_lr: float = 0.0003,
    ngram_adapt_decay: float = 0.001,
    packed_cache: bool = False,
    ngram_adapt_last_n_blocks: int = 3,
    lambda_schedule_spec: str = "",
    confidence_schedule_spec: str = "",
    order_lambdas_spec: str = "",
    max_windows: int = 0,
    log=print,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    if max_windows > 0:
        window_starts = window_starts[:max_windows]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ngram = build_ngram_cache(args.vocab_size, ngram_max_n, packed=packed_cache)
    if apply_mode not in {"improve_only", "always"}:
        raise ValueError(f"apply_mode must be one of improve_only/always, got {apply_mode}")
    ngram_applied = 0
    ngram_better = 0
    ngram_attempts = 0
    ngram_skipped = 0
    lambda_schedule = parse_lambda_schedule(lambda_schedule_spec, ngram_lambda)
    confidence_schedule = parse_confidence_schedule(confidence_schedule_spec, confidence_threshold)
    static_order_lambdas = parse_order_lambdas(order_lambdas_spec, ngram_max_n, ngram_lambda)
    ngram_adapt_optimizer = None
    global_weights = None

    if ngram_adapt_enabled:
        base_model.train()
        num_layers = len(base_model.blocks)
        update_params = []
        if ngram_adapt_last_n_blocks > 0:
            target_layers = list(range(max(0, num_layers - ngram_adapt_last_n_blocks), num_layers))
        else:
            target_layers = list(range(num_layers))
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
        f"confidence_threshold={confidence_threshold} gate_mode={gate_mode} min_count={min_count} "
        f"apply_mode={apply_mode} "
        f"adapt={int(ngram_adapt_enabled)} packed={int(packed_cache)} "
        f"lambda_schedule={format_lambda_schedule(lambda_schedule)} "
        f"confidence_schedule={format_confidence_schedule(confidence_schedule)} "
        f"order_lambdas={format_order_lambdas(static_order_lambdas)}"
    )
    for bi in range(0, len(window_starts), batch_seqs):
        batch_ws = window_starts[bi : bi + batch_seqs]
        bsz = len(batch_ws)
        batch_progress = bi / max(len(window_starts), 1)
        batch_lambda = lambda_for_progress(lambda_schedule, batch_progress)
        batch_confidence_threshold = confidence_for_progress(confidence_schedule, batch_progress)
        log_conf_thresh = gate_value_to_log_threshold(batch_confidence_threshold)
        if order_lambdas_spec.strip():
            batch_order_lambdas = static_order_lambdas
        else:
            batch_order_lambdas = {n: batch_lambda for n in range(2, ngram_max_n + 1)}
        batch_order_log_lambdas = {n: (math.log(1.0 - lam), math.log(lam)) for n, lam in batch_order_lambdas.items()}
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
                target_logps: list[float] = []
                n_scored = wlen - s
                for t_off in range(n_scored):
                    t_idx = s + t_off
                    tgt_tok = y_cpu[t_off]
                    target_lp = float(log_sm[t_off, tgt_tok].item())
                    gate_lp = max_logp_cpu[t_off] if gate_mode == "max" else target_lp
                    if gate_lp > log_conf_thresh:
                        ngram_skipped += 1
                        continue
                    uncertain_indices.append(t_off)
                    ctx_start = max(0, t_idx - ngram_max_n + 1)
                    prev_contexts.append(x_cpu[ctx_start : t_idx + 1])
                    targets.append(tgt_tok)
                    target_logps.append(target_lp)

                if uncertain_indices:
                    ng_lookups = [ngram.lookup_target(ctx, tgt, min_count=min_count) for ctx, tgt in zip(prev_contexts, targets)]
                    for j, t_off in enumerate(uncertain_indices):
                        ng_lookup = ng_lookups[j]
                        if ng_lookup is None:
                            continue
                        ng_lp, ng_order, _ = ng_lookup
                        ngram_attempts += 1
                        model_lp = target_logps[j]
                        log_1_minus_lam, log_lam = batch_order_log_lambdas[ng_order]
                        a = log_1_minus_lam + model_lp
                        b = log_lam + ng_lp
                        mixed_lp = max(a, b) + math.log1p(math.exp(-abs(a - b)))
                        new_nll = -mixed_lp
                        old_nll = scored_nll[t_off].item()
                        if apply_mode == "always" or new_nll < old_nll:
                            scored_nll[t_off] = new_nll
                            ngram_applied += 1
                        if new_nll < old_nll:
                            ngram_better += 1

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
            apply_rate = ngram_applied / max(ngram_attempts, 1) * 100
            better_rate = ngram_better / max(ngram_attempts, 1) * 100
            skip = ngram_skipped / max(ngram_skipped + ngram_attempts + 1, 1) * 100
            suffix = " +ngram_adapt" if ngram_adapt_enabled else ""
            log(
                f"  ngram [{pct:5.1f}%] bpb={rb:.6f} apply={apply_rate:.1f}% better={better_rate:.1f}% "
                f"skip={skip:.0f}% conf={batch_confidence_threshold:.2f} lam={batch_lambda:.3f} "
                f"mode={apply_mode}{suffix}"
            )

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    log(f"  ngram: {ngram_applied}/{ngram_attempts} applied, {ngram_better} better, {ngram_skipped} skipped")
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_hashed_ngram(
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
    ngram_lambda: float = 0.20,
    ngram_max_n: int = 5,
    min_count: int = 2,
    hashed_buckets: int = 4_194_304,
    log=print,
) -> tuple[float, float]:
    if ngram_max_n < 2:
        raise ValueError(f"hashed ngram requires max_n >= 2, got {ngram_max_n}")
    if hashed_buckets < 1024 or (hashed_buckets & (hashed_buckets - 1)) != 0:
        raise ValueError(f"hashed_buckets must be a power of two >=1024, got {hashed_buckets}")
    if not (0.0 <= ngram_lambda <= 1.0):
        raise ValueError(f"ngram_lambda must be in [0,1], got {ngram_lambda}")

    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    val_np = val_tokens.cpu().numpy()
    ctx_table = np.zeros((hashed_buckets,), dtype=np.uint32)
    full_table = np.zeros((hashed_buckets,), dtype=np.uint32)
    primes = np.array(
        [np.uint64(36313), np.uint64(27191), np.uint64(51647), np.uint64(81929), np.uint64(131071)],
        dtype=np.uint64,
    )

    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    log(
        "hashed_ngram_eval:start "
        f"stride={stride} alpha={ngram_lambda} order={ngram_max_n} "
        f"min_count={min_count} buckets={hashed_buckets}"
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
            seg_len = wlen - s
            if seg_len <= 0:
                continue

            seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
            seg_model_p = np.exp(-seg_nll)
            token_positions = np.arange(ws + s, ws + wlen, dtype=np.int64)
            valid = token_positions >= (ngram_max_n - 1)
            if valid.any():
                valid_positions = token_positions[valid]
                ctx_key, full_key = build_hashed_ngram_keys(
                    val_np,
                    valid_positions,
                    order=ngram_max_n,
                    buckets=hashed_buckets,
                    primes=primes,
                )
                ctx_counts = ctx_table[ctx_key].astype(np.float64)
                full_counts = full_table[full_key].astype(np.float64)
                can_mix = ctx_counts >= float(min_count)
                if can_mix.any():
                    p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                    p_ng = np.clip(p_ng, 0.0, 1.0)
                    v_idx = np.nonzero(valid)[0]
                    mixed = (1.0 - ngram_lambda) * seg_model_p[v_idx] + ngram_lambda * p_ng
                    seg_model_p[v_idx[can_mix]] = mixed[can_mix]
                seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                np.add.at(ctx_table, ctx_key, 1)
                np.add.at(full_table, full_key, 1)

            scored_nll = torch.from_numpy(seg_nll).to(device=device, dtype=torch.float64)
            loss_sum += scored_nll.sum()
            token_count += float(seg_len)
            tgt = y_batch[i, s:wlen]
            prev = x_batch[i, s:wlen]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count += tb.sum()

        if bi % (batch_seqs * 200) == 0 and bi > 0 and token_count.item() > 0:
            rl = loss_sum.item() / token_count.item()
            rb = (rl / math.log(2.0)) * (token_count.item() / max(byte_count.item(), 1))
            pct = 100.0 * (bi + bsz) / max(len(window_starts), 1)
            log(f"  hashed_ngram [{pct:5.1f}%] bpb={rb:.6f} alpha={ngram_lambda:.3f} min_count={min_count}")

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
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
    parser.add_argument("--gate-mode", choices=["max", "target"], default="max")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--apply-mode", choices=["improve_only", "always"], default="improve_only")
    parser.add_argument("--ngram-adapt-enabled", action="store_true")
    parser.add_argument("--ngram-adapt-lr", type=float, default=0.0003)
    parser.add_argument("--ngram-adapt-decay", type=float, default=0.001)
    parser.add_argument("--ngram-adapt-last-n-blocks", type=int, default=3)
    parser.add_argument("--packed-cache", action="store_true")
    parser.add_argument("--cache-kind", choices=["exact", "hashed"], default="exact")
    parser.add_argument("--hashed-buckets", type=int, default=4_194_304)
    parser.add_argument("--lambda-schedule", type=str, default="")
    parser.add_argument("--confidence-schedule", type=str, default="")
    parser.add_argument("--order-lambdas", type=str, default="")
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
                "gate_mode": args_ns.gate_mode,
                "min_count": args_ns.min_count,
                "apply_mode": args_ns.apply_mode,
                "ngram_adapt_enabled": int(args_ns.ngram_adapt_enabled),
                "ngram_adapt_lr": args_ns.ngram_adapt_lr,
                "ngram_adapt_decay": args_ns.ngram_adapt_decay,
                "ngram_adapt_last_n_blocks": args_ns.ngram_adapt_last_n_blocks,
                "packed_cache": int(args_ns.packed_cache),
                "cache_kind": args_ns.cache_kind,
                "hashed_buckets": args_ns.hashed_buckets,
                "confidence_schedule": args_ns.confidence_schedule,
                "order_lambdas": args_ns.order_lambdas,
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
    if args_ns.cache_kind == "hashed":
        val_loss, val_bpb = eval_val_hashed_ngram(
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
            min_count=args_ns.min_count,
            hashed_buckets=args_ns.hashed_buckets,
            log=log,
        )
    else:
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
            gate_mode=args_ns.gate_mode,
            min_count=args_ns.min_count,
            apply_mode=args_ns.apply_mode,
            ngram_adapt_enabled=args_ns.ngram_adapt_enabled,
            ngram_adapt_lr=args_ns.ngram_adapt_lr,
            ngram_adapt_decay=args_ns.ngram_adapt_decay,
            ngram_adapt_last_n_blocks=args_ns.ngram_adapt_last_n_blocks,
            packed_cache=args_ns.packed_cache,
            lambda_schedule_spec=args_ns.lambda_schedule,
            confidence_schedule_spec=args_ns.confidence_schedule,
            order_lambdas_spec=args_ns.order_lambdas,
            max_windows=args_ns.max_windows,
            log=log,
        )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"final_ngram_eval val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} stride:{args_ns.stride} eval_time:{elapsed_ms:.0f}ms")
    log(f"final_ngram_eval_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    if args_ns.cache_kind == "hashed":
        log(
            f"final_int6_sliding_window_ngram{args_ns.ngram_max_n} val_loss:{val_loss:.4f} "
            f"val_bpb:{val_bpb:.4f} eval_time:{elapsed_ms:.0f}ms"
        )
        log(
            f"final_int6_sliding_window_ngram{args_ns.ngram_max_n}_exact "
            f"val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}"
        )
    log(
        f"resume_ngram_eval:peak_memory_allocated_mib={torch.cuda.max_memory_allocated() // 1024 // 1024} "
        f"reserved_mib={torch.cuda.max_memory_reserved() // 1024 // 1024}"
    )


if __name__ == "__main__":
    main()
