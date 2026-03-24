#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import lzma
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish legal TTT eval from a saved Parameter Golf artifact.")
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
    parser.add_argument("--ttt-freeze-blocks", type=int, default=0)
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
    args.ttt_enabled = True
    args.ttt_batch_seqs = args_ns.batch_seqs
    args.bigram_vocab_size = args_ns.bigram_vocab_size
    args.ttt_freeze_blocks = args_ns.ttt_freeze_blocks
    args.eval_stride = 64
    args.extra_stride64_final_eval = False

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

    log("resume_ttt_eval:start")
    log(f"resume_ttt_eval:artifact_path={artifact_path}")
    log(f"resume_ttt_eval:template_path={template_path}")
    log(f"resume_ttt_eval:train_gpt_path={train_gpt_path}")
    log(f"resume_ttt_eval:device={device}")
    log(f"resume_ttt_eval:torch={torch.__version__}")
    log(
        json.dumps(
            {
                "ttt_lr": args.ttt_lr,
                "ttt_epochs": args.ttt_epochs,
                "ttt_chunk_tokens": args.ttt_chunk_tokens,
                "ttt_freeze_blocks": args.ttt_freeze_blocks,
                "ttt_momentum": args.ttt_momentum,
                "ttt_batch_seqs": args.ttt_batch_seqs,
                "ttt_grad_clip": args.ttt_grad_clip,
            },
            sort_keys=True,
        )
    )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = mod.load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

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
    ttt_loss, ttt_bpb = mod.eval_val_sliding_ttt(
        args,
        eval_model,
        rank=0,
        world_size=1,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.eval_stride,
        batch_seqs=args.ttt_batch_seqs,
        log0=log,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{elapsed_ms:.0f}ms")
    log(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    log(
        f"resume_ttt_eval:peak_memory_allocated_mib={torch.cuda.max_memory_allocated() // 1024 // 1024} "
        f"reserved_mib={torch.cuda.max_memory_reserved() // 1024 // 1024}"
    )


if __name__ == "__main__":
    main()
