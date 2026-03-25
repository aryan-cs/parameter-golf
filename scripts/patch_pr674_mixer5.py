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
    parser = argparse.ArgumentParser(description="Patch PR674 train_gpt.py with a PR688-style 5-expert online mixer.")
    parser.add_argument("train_gpt", type=Path)
    args = parser.parse_args()

    path = args.train_gpt
    text = path.read_text()

    text = replace_once(
        text,
        '    ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", 4_194_304))\n'
        '    ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", 0.0))\n'
        '    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))\n',
        '    ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", 4_194_304))\n'
        '    ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", 0.0))\n'
        '    ngram_mixer5_enabled = bool(int(os.environ.get("NGRAM_MIXER5_ENABLED", "0")))\n'
        '    ngram_mixer5_eta = float(os.environ.get("NGRAM_MIXER5_ETA", 0.10))\n'
        '    ngram_mixer5_neural_bias = float(os.environ.get("NGRAM_MIXER5_NEURAL_BIAS", 2.0))\n'
        '    ngram_mixer5_trigram_buckets = int(os.environ.get("NGRAM_MIXER5_TRIGRAM_BUCKETS", 65536))\n'
        '    ngram_mixer5_warmup_tokens = int(os.environ.get("NGRAM_MIXER5_WARMUP_TOKENS", 10000))\n'
        '    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))\n',
    )

    text = replace_once(
        text,
        "def maybe_torch_compile(obj, args: Hyperparameters):\n"
        "    if not args.compile_enabled:\n"
        "        return obj\n"
        "    return torch.compile(obj, dynamic=False, fullgraph=args.compile_fullgraph)\n",
        "def maybe_torch_compile(obj, args: Hyperparameters):\n"
        "    if not args.compile_enabled:\n"
        "        return obj\n"
        "    return torch.compile(obj, dynamic=False, fullgraph=args.compile_fullgraph)\n"
        "def _logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:\n"
        "    x_max = np.max(x, axis=axis, keepdims=True)\n"
        "    return np.squeeze(x_max, axis=axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))\n",
    )

    text = replace_once(
        text,
        "    loss_sum = 0.0\n"
        "    token_count = 0.0\n"
        "    byte_count = 0.0\n"
        "\n"
        "    base_model.eval()\n",
        "    loss_sum = 0.0\n"
        "    token_count = 0.0\n"
        "    byte_count = 0.0\n"
        "\n"
        "    mixer5_enabled = args.ngram_mixer5_enabled\n"
        "    mixer5_log_weights = np.zeros((5,), dtype=np.float64)\n"
        "    mixer5_log_weights[0] = args.ngram_mixer5_neural_bias\n"
        "    mixer5_uni_counts = np.zeros((args.vocab_size,), dtype=np.uint32)\n"
        "    mixer5_bi_counts = np.zeros((args.vocab_size, args.vocab_size), dtype=np.uint32)\n"
        "    mixer5_bi_row_totals = np.zeros((args.vocab_size,), dtype=np.uint32)\n"
        "    mixer5_tri_buckets = args.ngram_mixer5_trigram_buckets\n"
        "    mixer5_tri_mask = np.uint64(mixer5_tri_buckets - 1)\n"
        "    mixer5_tri_counts = np.zeros((mixer5_tri_buckets, args.vocab_size), dtype=np.uint32)\n"
        "    mixer5_tri_row_totals = np.zeros((mixer5_tri_buckets,), dtype=np.uint32)\n"
        "    mixer5_total_tokens = 0\n"
        "    mixer5_primes = np.array([np.uint64(36313), np.uint64(27191)], dtype=np.uint64)\n"
        "\n"
        "    base_model.eval()\n",
    )

    text = replace_once(
        text,
        "    t0 = time.perf_counter()\n"
        "    deadline = (t0 + max_seconds) if max_seconds > 0.0 else None\n",
        "    t0 = time.perf_counter()\n"
        "    if mixer5_enabled:\n"
        '        print(f"ngram_eval:mixer5 enabled eta={args.ngram_mixer5_eta:.4f} neural_bias={args.ngram_mixer5_neural_bias:.3f} trigram_buckets={args.ngram_mixer5_trigram_buckets} warmup_tokens={args.ngram_mixer5_warmup_tokens}", flush=True)\n'
        "    deadline = (t0 + max_seconds) if max_seconds > 0.0 else None\n",
    )

    text = replace_once(
        text,
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16):\n"
        "                logits = compiled_logits(x_batch)\n"
        "            nll = F.cross_entropy(\n"
        "                logits.reshape(-1, logits.size(-1)).float(),\n"
        "                y_batch.reshape(-1),\n"
        "                reduction=\"none\",\n"
        "            ).reshape(bsz, seq_len)\n"
        "\n"
        "            for i, ws in enumerate(batch_ws):\n",
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16):\n"
        "                logits = compiled_logits(x_batch)\n"
        "            logits_f = logits.float()\n"
        "            log_sm = F.log_softmax(logits_f, dim=-1)\n"
        "            nll = -log_sm.gather(2, y_batch.unsqueeze(2)).squeeze(2)\n"
        "\n"
        "            for i, ws in enumerate(batch_ws):\n",
    )

    text = replace_once(
        text,
        "                seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()\n"
        "                seg_model_p = np.exp(-seg_nll)\n"
        "\n"
        "                global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)\n"
        "                valid = global_j >= (order - 1)\n"
        "                if valid.any():\n"
        "                    v_idx = np.nonzero(valid)[0]\n"
        "                    jv = global_j[v_idx]\n"
        "\n"
        "                    ctx_hash = np.zeros((len(jv),), dtype=np.uint64)\n"
        "                    ctx_width = order - 1\n"
        "                    for k in range(ctx_width):\n"
        "                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)\n"
        "                        ctx_hash ^= tok * primes[k % len(primes)]\n"
        "                    ctx_key = (ctx_hash & mask).astype(np.int64)\n"
        "\n"
        "                    tgt_np = val_np[jv].astype(np.uint64)\n"
        "                    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)\n"
        "\n"
        "                    ctx_counts = ctx_table[ctx_key].astype(np.float64)\n"
        "                    full_counts = full_table[full_key].astype(np.float64)\n"
        "                    can_mix = ctx_counts >= float(min_count)\n"
        "                    if can_mix.any():\n"
        "                        # Collision-safe estimate: ensure n-gram probability stays in [0, 1].\n"
        "                        # With hashed sketches, full_counts can exceed ctx_counts due collisions.\n"
        "                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)\n"
        "                        p_ng = np.clip(p_ng, 0.0, 1.0)\n"
        "                        mixed = (1.0 - alpha) * seg_model_p[v_idx] + alpha * p_ng\n"
        "                        seg_model_p[v_idx[can_mix]] = mixed[can_mix]\n"
        "                    seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))\n"
        "\n"
        "                    # Score-first legality: update cache only after segment scoring.\n"
        "                    np.add.at(ctx_table, ctx_key, 1)\n"
        "                    np.add.at(full_table, full_key, 1)\n",
        "                seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()\n"
        "                seg_log_sm = log_sm[i, s:wlen].cpu().numpy()\n"
        "                seg_model_p = np.exp(-seg_nll)\n"
        "\n"
        "                global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)\n"
        "                valid = global_j >= (order - 1)\n"
        "                if valid.any():\n"
        "                    v_idx = np.nonzero(valid)[0]\n"
        "                    jv = global_j[v_idx]\n"
        "\n"
        "                    ctx_hash = np.zeros((len(jv),), dtype=np.uint64)\n"
        "                    ctx_width = order - 1\n"
        "                    for k in range(ctx_width):\n"
        "                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)\n"
        "                        ctx_hash ^= tok * primes[k % len(primes)]\n"
        "                    ctx_key = (ctx_hash & mask).astype(np.int64)\n"
        "\n"
        "                    tgt_np = val_np[jv].astype(np.uint64)\n"
        "                    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)\n"
        "\n"
        "                    ctx_counts = ctx_table[ctx_key].astype(np.float64)\n"
        "                    full_counts = full_table[full_key].astype(np.float64)\n"
        "                    can_mix = ctx_counts >= float(min_count)\n"
        "                    if can_mix.any():\n"
        "                        # Collision-safe estimate: ensure n-gram probability stays in [0, 1].\n"
        "                        # With hashed sketches, full_counts can exceed ctx_counts due collisions.\n"
        "                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)\n"
        "                        p_ng = np.clip(p_ng, 0.0, 1.0)\n"
        "                        if mixer5_enabled and mixer5_total_tokens >= args.ngram_mixer5_warmup_tokens:\n"
        "                            mix_idx = v_idx\n"
        "                            model_probs = np.clip(seg_model_p[mix_idx], 1e-12, 1.0)\n"
        "                            target_ids = val_np[jv].astype(np.int64)\n"
        "                            prev_ids = val_np[jv - 1].astype(np.int64)\n"
        "                            prev2_ids = val_np[jv - 2].astype(np.uint64)\n"
        "                            uni_probs = (mixer5_uni_counts[target_ids].astype(np.float64) + 0.1) / (float(mixer5_total_tokens) + 0.1 * args.vocab_size)\n"
        "                            bi_totals = mixer5_bi_row_totals[prev_ids].astype(np.float64)\n"
        "                            bi_counts = mixer5_bi_counts[prev_ids, target_ids].astype(np.float64)\n"
        "                            bi_probs = (bi_counts + 0.1) / (bi_totals + 0.1 * args.vocab_size)\n"
        "                            tri_ctx = ((prev2_ids * mixer5_primes[0]) ^ (prev_ids.astype(np.uint64) * mixer5_primes[1])) & mixer5_tri_mask\n"
        "                            tri_ctx_i = tri_ctx.astype(np.int64)\n"
        "                            tri_totals = mixer5_tri_row_totals[tri_ctx_i].astype(np.float64)\n"
        "                            tri_counts = mixer5_tri_counts[tri_ctx_i, target_ids].astype(np.float64)\n"
        "                            tri_probs = (tri_counts + 0.01) / (tri_totals + 0.01 * args.vocab_size)\n"
        "                            hash_probs = np.where(can_mix, p_ng, model_probs)\n"
        "                            entropy_nll = -(np.exp(seg_log_sm[mix_idx]) * seg_log_sm[mix_idx]).sum(axis=-1)\n"
        "                            expert_nll = np.stack([\n"
        "                                -np.log(model_probs),\n"
        "                                -np.log(np.clip(uni_probs, 1e-12, 1.0)),\n"
        "                                -np.log(np.clip(bi_probs, 1e-12, 1.0)),\n"
        "                                -np.log(np.clip(tri_probs, 1e-12, 1.0)),\n"
        "                                -np.log(np.clip(hash_probs, 1e-12, 1.0)),\n"
        "                            ], axis=-1)\n"
        "                            norm_log_w = mixer5_log_weights - np.logaddexp.reduce(mixer5_log_weights)\n"
        "                            mixed_lp = _logsumexp_np(norm_log_w[None, :] - expert_nll, axis=-1)\n"
        "                            seg_nll[mix_idx] = -mixed_lp\n"
        "                            mixer5_log_weights -= args.ngram_mixer5_eta * expert_nll.mean(axis=0)\n"
        "                        else:\n"
        "                            mixed = (1.0 - alpha) * seg_model_p[v_idx] + alpha * p_ng\n"
        "                            seg_model_p[v_idx[can_mix]] = mixed[can_mix]\n"
        "                            seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))\n"
        "\n"
        "                    # Score-first legality: update cache only after segment scoring.\n"
        "                    np.add.at(ctx_table, ctx_key, 1)\n"
        "                    np.add.at(full_table, full_key, 1)\n"
        "\n"
        "                scored_targets = val_np[ws + s + 1 : ws + wlen + 1].astype(np.int64)\n"
        "                mixer5_total_tokens += int(scored_targets.size)\n"
        "                if scored_targets.size > 0:\n"
        "                    np.add.at(mixer5_uni_counts, scored_targets, 1)\n"
        "                    scored_prev = val_np[ws + s : ws + wlen].astype(np.int64)\n"
        "                    np.add.at(mixer5_bi_counts.reshape(-1), scored_prev * args.vocab_size + scored_targets, 1)\n"
        "                    np.add.at(mixer5_bi_row_totals, scored_prev, 1)\n"
        "                    if ws + s >= 1:\n"
        "                        scored_prev2 = val_np[ws + s - 1 : ws + wlen - 1].astype(np.uint64)\n"
        "                        tri_ctx = ((scored_prev2 * mixer5_primes[0]) ^ (scored_prev.astype(np.uint64) * mixer5_primes[1])) & mixer5_tri_mask\n"
        "                        np.add.at(mixer5_tri_counts.reshape(-1), tri_ctx.astype(np.int64) * args.vocab_size + scored_targets, 1)\n"
        "                        np.add.at(mixer5_tri_row_totals, tri_ctx.astype(np.int64), 1)\n",
    )

    text = replace_once(
        text,
        "    if cutoff_hit:\n"
        "        elapsed = time.perf_counter() - t0\n"
        "        print(\n"
        "            f\"ngram_eval:cutoff max_seconds={max_seconds:.1f} \"\n"
        "            f\"coverage={coverage*100:.2f}% elapsed={elapsed:.0f}s\",\n"
        "            flush=True,\n"
        "        )\n"
        "\n"
        "    val_loss = loss_sum / max(token_count, 1.0)\n",
        "    if cutoff_hit:\n"
        "        elapsed = time.perf_counter() - t0\n"
        "        print(\n"
        "            f\"ngram_eval:cutoff max_seconds={max_seconds:.1f} \"\n"
        "            f\"coverage={coverage*100:.2f}% elapsed={elapsed:.0f}s\",\n"
        "            flush=True,\n"
        "        )\n"
        "    if mixer5_enabled:\n"
        "        final_log_w = mixer5_log_weights - np.logaddexp.reduce(mixer5_log_weights)\n"
        '        print(f"ngram_eval:mixer5_final neural={np.exp(final_log_w[0]):.6f} unigram={np.exp(final_log_w[1]):.6f} bigram={np.exp(final_log_w[2]):.6f} trigram={np.exp(final_log_w[3]):.6f} hash5={np.exp(final_log_w[4]):.6f}", flush=True)\n'
        "\n"
        "    val_loss = loss_sum / max(token_count, 1.0)\n",
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
