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
    parser = argparse.ArgumentParser(description="Patch PR674 train_gpt.py with a legal Hedge-style adaptive n-gram mixer.")
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
        '    ngram_hedge_enabled = bool(int(os.environ.get("NGRAM_HEDGE_ENABLED", "0")))\n'
        '    ngram_hedge_eta = float(os.environ.get("NGRAM_HEDGE_ETA", 0.10))\n'
        '    ngram_hedge_neural_bias = float(os.environ.get("NGRAM_HEDGE_NEURAL_BIAS", 2.0))\n'
        '    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))\n',
    )

    text = replace_once(
        text,
        "    base_model.eval()\n"
        "    compiled_logits = maybe_torch_compile(base_model.forward_logits, args)\n"
        "    t0 = time.perf_counter()\n",
        "    hedge_enabled = args.ngram_hedge_enabled\n"
        "    hedge_eta = args.ngram_hedge_eta\n"
        "    hedge_log_weights = np.array([args.ngram_hedge_neural_bias, 0.0], dtype=np.float64)\n"
        "\n"
        "    base_model.eval()\n"
        "    compiled_logits = maybe_torch_compile(base_model.forward_logits, args)\n"
        "    t0 = time.perf_counter()\n"
        "    if hedge_enabled:\n"
        '        print(f"ngram_eval:hedge enabled eta={hedge_eta:.4f} neural_bias={args.ngram_hedge_neural_bias:.3f}", flush=True)\n',
    )

    text = replace_once(
        text,
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
        "                    seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))\n",
        "                    ctx_counts = ctx_table[ctx_key].astype(np.float64)\n"
        "                    full_counts = full_table[full_key].astype(np.float64)\n"
        "                    can_mix = ctx_counts >= float(min_count)\n"
        "                    if can_mix.any():\n"
        "                        # Collision-safe estimate: ensure n-gram probability stays in [0, 1].\n"
        "                        # With hashed sketches, full_counts can exceed ctx_counts due collisions.\n"
        "                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)\n"
        "                        p_ng = np.clip(p_ng, 0.0, 1.0)\n"
        "                        mix_idx = v_idx[can_mix]\n"
        "                        if hedge_enabled:\n"
        "                            p_model = np.clip(seg_model_p[mix_idx], 1e-12, 1.0)\n"
        "                            p_ng_mix = np.clip(p_ng[can_mix], 1e-12, 1.0)\n"
        "                            norm = np.logaddexp.reduce(hedge_log_weights)\n"
        "                            log_mix = np.logaddexp(\n"
        "                                hedge_log_weights[0] - norm + np.log(p_model),\n"
        "                                hedge_log_weights[1] - norm + np.log(p_ng_mix),\n"
        "                            )\n"
        "                            seg_model_p[mix_idx] = np.exp(log_mix)\n"
        "                            expert_mean_loss = np.array(\n"
        "                                [\n"
        "                                    float((-np.log(p_model)).mean()),\n"
        "                                    float((-np.log(p_ng_mix)).mean()),\n"
        "                                ],\n"
        "                                dtype=np.float64,\n"
        "                            )\n"
        "                            hedge_log_weights -= hedge_eta * expert_mean_loss\n"
        "                        else:\n"
        "                            mixed = (1.0 - alpha) * seg_model_p[v_idx] + alpha * p_ng\n"
        "                            seg_model_p[mix_idx] = mixed[can_mix]\n"
        "                    seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))\n",
    )

    text = replace_once(
        text,
        "    val_loss = loss_sum / max(token_count, 1.0)\n",
        "    if hedge_enabled:\n"
        "        final_log_w = hedge_log_weights - np.logaddexp.reduce(hedge_log_weights)\n"
        '        print(f"ngram_eval:hedge_final neural={np.exp(final_log_w[0]):.6f} ngram={np.exp(final_log_w[1]):.6f}", flush=True)\n'
        "    val_loss = loss_sum / max(token_count, 1.0)\n",
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
