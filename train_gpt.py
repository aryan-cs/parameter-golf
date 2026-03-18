#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import random
import tempfile
import time
import zlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import sentencepiece as spm
except ModuleNotFoundError:
    spm = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


LOG2_E = math.log2(math.e)
OFFICIAL_DATAFILE_MAGIC = 20240520
OFFICIAL_DATAFILE_HEADER_INTS = 256
OFFICIAL_DATAFILE_HEADER_BYTES = OFFICIAL_DATAFILE_HEADER_INTS * 4


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    run_id: str
    data_path: str
    val_data_path: str
    token_dtype: str | None
    vocab_size: int | None
    d_model: int
    n_heads: int
    d_ff: int
    n_loops: int
    seq_len: int
    train_batch_tokens: int
    val_batch_tokens: int
    val_steps: int
    val_loss_every: int
    max_steps: int
    max_wallclock_seconds: int
    avg_bytes_per_token: float | None
    tokenizer_path: str
    muon_lr: float
    adamw_lr: float
    weight_decay: float
    warmup_steps: int
    cooldown_fraction: float
    qat_start_fraction: float
    grad_clip: float
    seed: int
    device: str
    compile_model: bool
    use_smear: bool
    artifact_path: str
    stats_path: str

    @classmethod
    def from_env(cls) -> "Config":
        d_model = env_int("D_MODEL", 256)
        return cls(
            run_id=os.environ.get("RUN_ID", "dev_smoke"),
            data_path=os.environ.get("DATA_PATH", ""),
            val_data_path=os.environ.get("VAL_DATA_PATH", ""),
            token_dtype=os.environ.get("TOKEN_DTYPE") or None,
            vocab_size=env_int_optional("VOCAB_SIZE"),
            d_model=d_model,
            n_heads=env_int("N_HEADS", 8),
            d_ff=env_int("D_FF", max(4, int((8 * d_model) / 3))),
            n_loops=env_int("N_LOOPS", 4),
            seq_len=env_int("SEQ_LEN", 128),
            train_batch_tokens=env_int("TRAIN_BATCH_TOKENS", 8192),
            val_batch_tokens=env_int("VAL_BATCH_TOKENS", 8192),
            val_steps=env_int("VAL_STEPS", 4),
            val_loss_every=env_int("VAL_LOSS_EVERY", 10),
            max_steps=env_int("MAX_STEPS", 20),
            max_wallclock_seconds=env_int("MAX_WALLCLOCK_SECONDS", 0),
            avg_bytes_per_token=env_float_optional("AVG_BYTES_PER_TOKEN"),
            tokenizer_path=os.environ.get("TOKENIZER_PATH", ""),
            muon_lr=env_float("MUON_LR", 0.02),
            adamw_lr=env_float("ADAMW_LR", 3e-4),
            weight_decay=env_float("WEIGHT_DECAY", 0.1),
            warmup_steps=env_int("WARMUP_STEPS", 20),
            cooldown_fraction=env_float("COOLDOWN_FRACTION", 0.3),
            qat_start_fraction=env_float("QAT_START_FRACTION", 0.6),
            grad_clip=env_float("GRAD_CLIP", 1.0),
            seed=env_int("SEED", 1337),
            device=os.environ.get("DEVICE", ""),
            compile_model=env_bool("COMPILE_MODEL", False),
            use_smear=env_bool("USE_SMEAR", True),
            artifact_path=os.environ.get("ARTIFACT_PATH", ""),
            stats_path=os.environ.get("STATS_PATH", ""),
        )


if torch is not None:
    if hasattr(nn, "RMSNorm"):
        RMSNorm = nn.RMSNorm
    else:
        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x):
                rms = x.pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(rms + self.eps)
                return x * self.weight


    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int) -> None:
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("sin_cached", torch.empty(0), persistent=False)
            self.cached_seq_len = 0
            self._build_cache(max_seq_len)

        def _build_cache(self, seq_len: int) -> None:
            t = torch.arange(seq_len, device=self.inv_freq.device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
            self.cached_seq_len = seq_len

        def forward(self, seq_len: int, dtype):
            if seq_len > self.cached_seq_len:
                self._build_cache(seq_len)
            cos = self.cos_cached[:, :, :seq_len, :].to(dtype=dtype)
            sin = self.sin_cached[:, :, :seq_len, :].to(dtype=dtype)
            return cos, sin


    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


    def apply_rotary(q, k, cos, sin):
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


    def fake_quantize_int8(tensor):
        scale = tensor.detach().abs().max() / 127.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = (tensor / scale).round().clamp(-127, 127)
        return quantized * scale


    class QATLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.qat_enabled = False

        def enable_qat(self) -> None:
            self.qat_enabled = True

        def forward(self, x):
            weight = fake_quantize_int8(self.linear.weight) if self.qat_enabled else self.linear.weight
            return F.linear(x, weight, self.linear.bias)


    class SmearBlock(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.gate = nn.Parameter(torch.zeros(d_model))

        def forward(self, x):
            x_prev = torch.roll(x, shifts=1, dims=1)
            x_prev[:, 0, :] = 0
            return x + torch.sigmoid(self.gate).view(1, 1, -1) * x_prev


    class SharedTransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("D_MODEL must be divisible by N_HEADS")
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.qkv = QATLinear(d_model, 3 * d_model, bias=False)
            self.out_proj = QATLinear(d_model, d_model, bias=False)
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
            self.ff_gate = QATLinear(d_model, d_ff, bias=False)
            self.ff_up = QATLinear(d_model, d_ff, bias=False)
            self.ff_down = QATLinear(d_ff, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def attention(self, x, cos, sin):
            batch_size, seq_len, channels = x.shape
            qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q, k = apply_rotary(q, k, cos, sin)
            dropout_p = self.dropout.p if self.training else 0.0
            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
            else:
                scale = 1.0 / math.sqrt(self.d_head)
                scores = (q @ k.transpose(-2, -1)) * scale
                mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float("-inf"))
                probs = scores.softmax(dim=-1)
                out = probs @ v
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, channels)
            return self.out_proj(out)

        def ffn(self, x):
            return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))

        def forward(self, x, cos, sin):
            x = x + self.dropout(self.attention(self.norm1(x), cos, sin))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x


    class LoopedTransformer(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            n_loops: int,
            max_seq_len: int,
            use_smear: bool,
        ) -> None:
            super().__init__()
            self.n_loops = n_loops
            self.embed = nn.Embedding(vocab_size, d_model)
            self.smear = SmearBlock(d_model) if use_smear else nn.Identity()
            self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
            self.block = SharedTransformerBlock(d_model, n_heads, d_ff)
            self.loop_embeddings = nn.Embedding(n_loops, d_model)
            self.final_norm = RMSNorm(d_model)
            self.head = QATLinear(d_model, vocab_size, bias=False)
            self._init_weights()

        def _init_weights(self) -> None:
            nn.init.normal_(self.embed.weight, std=0.02)
            nn.init.normal_(self.loop_embeddings.weight, std=0.02)
            nn.init.normal_(self.head.linear.weight, std=0.02)
            for name, param in self.block.named_parameters():
                if "weight" in name and param.ndim == 2:
                    nn.init.normal_(param, std=0.02 / math.sqrt(2 * self.n_loops))

        def forward(self, input_ids):
            _, seq_len = input_ids.shape
            x = self.embed(input_ids)
            x = self.smear(x)
            cos, sin = self.rope(seq_len, x.dtype)
            loop_ids = torch.arange(self.n_loops, device=input_ids.device)
            loop_emb = self.loop_embeddings(loop_ids)
            for loop_idx in range(self.n_loops):
                x = x + loop_emb[loop_idx].view(1, 1, -1)
                x = self.block(x, cos, sin)
            x = self.final_norm(x)
            return self.head(x)


    def enable_qat(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, QATLinear):
                module.enable_qat()


    def zeropower_via_newtonschulz5(grad, steps: int = 5, eps: float = 1e-7):
        if grad.ndim != 2:
            raise ValueError("Muon only supports 2D gradients")
        a, b, c = 3.4445, -4.7750, 2.0315
        work = grad.float() / (grad.norm().float() + eps)
        transposed = work.size(0) > work.size(1)
        if transposed:
            work = work.t()
        for _ in range(steps):
            gram = work @ work.t()
            work = a * work + b * gram @ work + c * gram @ gram @ work
        if transposed:
            work = work.t()
        return work.type_as(grad)


    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if grad.ndim != 2:
                        continue
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = grad + momentum * buf if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update = update * max(1.0, param.size(0) / max(1, param.size(1))) ** 0.5
                    param.add_(update, alpha=-lr)
            return loss


class SyntheticTokenBatcher:
    def __init__(self, vocab_size: int, seq_len: int, seed: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.generator = random.Random(seed)

    def next_batch(self, batch_size: int, device):
        if torch is None:
            raise RuntimeError("torch is required to build token batches")
        batch = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.seq_len + 1),
            dtype=torch.long,
        )
        return batch.to(device)


class PackedTokenBatcher:
    def __init__(self, arrays, seq_len: int, seed: int) -> None:
        self.arrays = arrays
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def next_batch(self, batch_size: int, device):
        if torch is None or np is None:
            raise RuntimeError("torch and numpy are required for packed token loading")
        batch = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        for row in range(batch_size):
            arr = self.arrays[self.rng.randrange(len(self.arrays))]
            start = self.rng.randrange(0, len(arr) - self.seq_len - 1)
            batch[row] = np.asarray(arr[start:start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(batch).to(device)


class SentencePieceBPBHelper:
    def __init__(self, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut) -> None:
        self.base_bytes_lut = base_bytes_lut
        self.has_leading_space_lut = has_leading_space_lut
        self.is_boundary_token_lut = is_boundary_token_lut


def get_lr(step: int, total_steps: int, max_lr: float, warmup_steps: int, cooldown_fraction: float) -> float:
    if total_steps <= 1:
        return max_lr
    cooldown_start = max(warmup_steps + 1, int(total_steps * (1.0 - cooldown_fraction)))
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step < cooldown_start:
        progress = (step - warmup_steps) / max(1, cooldown_start - warmup_steps)
        return max_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    progress = (step - cooldown_start) / max(1, total_steps - cooldown_start)
    return max_lr * 0.1 * max(0.0, 1.0 - progress)


def resolve_device(requested: str):
    if torch is None:
        raise RuntimeError("torch is required to resolve a device")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_token_arrays(data_path: str, token_dtype: str, min_length: int, split: str = "any"):
    if np is None:
        raise RuntimeError("numpy is required to load token data from disk")
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {data_path}")
    if root.is_file():
        candidates = [root]
    else:
        candidates = []
        if split in {"train", "val"}:
            split_pattern = f"fineweb_{split}_*.bin"
            candidates.extend(sorted(root.glob(split_pattern)))
            if not candidates:
                candidates.extend(sorted(root.rglob(split_pattern)))
        if not candidates:
            candidates = sorted(root.rglob("*.npy")) + sorted(root.rglob("*.bin"))
    arrays = []
    dtype = np.dtype(token_dtype)
    for candidate in candidates:
        if candidate.suffix == ".npy":
            arr = np.load(candidate, mmap_mode="r")
        else:
            arr = open_token_bin(candidate, dtype)
        arr = arr.reshape(-1)
        if arr.shape[0] > min_length:
            arrays.append(arr)
    if not arrays:
        raise FileNotFoundError(f"No token arrays longer than {min_length} found under {data_path}")
    return arrays


def open_token_bin(path: Path, dtype):
    if np is None:
        raise RuntimeError("numpy is required to load token data from disk")
    file_size = path.stat().st_size
    if file_size >= OFFICIAL_DATAFILE_HEADER_BYTES:
        with path.open("rb") as handle:
            header = np.frombuffer(handle.read(OFFICIAL_DATAFILE_HEADER_BYTES), dtype="<i4", count=OFFICIAL_DATAFILE_HEADER_INTS)
        if header.size >= 3 and int(header[0]) == OFFICIAL_DATAFILE_MAGIC:
            token_count = int(header[2])
            if token_count < 0:
                raise ValueError(f"Negative token count in official datafile header: {path}")
            payload_bytes = file_size - OFFICIAL_DATAFILE_HEADER_BYTES
            expected_bytes = token_count * np.dtype(dtype).itemsize
            if expected_bytes > payload_bytes:
                raise ValueError(
                    f"Header token count exceeds payload size for {path}: "
                    f"expected_bytes={expected_bytes} payload_bytes={payload_bytes}"
                )
            return np.memmap(path, dtype=dtype, mode="r", offset=OFFICIAL_DATAFILE_HEADER_BYTES, shape=(token_count,))
    return np.memmap(path, dtype=dtype, mode="r")


def read_split_metadata(data_path: str):
    if not data_path:
        return None
    path = Path(data_path)
    metadata_path = (path if path.is_dir() else path.parent) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_avg_bytes_per_token(cfg: Config, val_path: str) -> tuple[float, str]:
    if cfg.avg_bytes_per_token is not None:
        return cfg.avg_bytes_per_token, "env"

    for candidate in (val_path, cfg.data_path):
        metadata = read_split_metadata(candidate)
        if metadata is None:
            continue
        avg = metadata.get("avg_bytes_per_token")
        if avg is not None:
            return float(avg), f"metadata:{Path(candidate).resolve()}"

    return 3.5, "default"


def resolve_vocab_size(cfg: Config, val_path: str) -> tuple[int, str]:
    if cfg.vocab_size is not None:
        return cfg.vocab_size, "env"

    for candidate in (val_path, cfg.data_path):
        metadata = read_split_metadata(candidate)
        if metadata is None:
            continue
        vocab_size = metadata.get("vocab_size")
        if vocab_size is not None:
            return int(vocab_size), f"metadata:{Path(candidate).resolve()}"

    return 32768, "default"


def resolve_token_dtype(cfg: Config, val_path: str) -> tuple[str, str]:
    if cfg.token_dtype:
        return cfg.token_dtype, "env"

    for candidate in (val_path, cfg.data_path):
        metadata = read_split_metadata(candidate)
        if metadata is None:
            continue
        token_dtype = metadata.get("token_dtype")
        if token_dtype:
            return str(token_dtype), f"metadata:{Path(candidate).resolve()}"

    return "uint16", "default"


def build_sentencepiece_bpb_helper(tokenizer_path: str, vocab_size: int) -> SentencePieceBPBHelper:
    if spm is None:
        raise RuntimeError("sentencepiece is required for exact tokenizer-aware val_bpb")
    if np is None:
        raise RuntimeError("numpy is required for exact tokenizer-aware val_bpb")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return SentencePieceBPBHelper(base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)


def build_batcher(cfg: Config, data_path: str, seed: int, split: str):
    if not data_path:
        return SyntheticTokenBatcher(cfg.vocab_size, cfg.seq_len, seed), "synthetic", None
    arrays = discover_token_arrays(data_path, cfg.token_dtype, cfg.seq_len + 1, split=split)
    return PackedTokenBatcher(arrays, cfg.seq_len, seed), f"{len(arrays)} shard(s)", arrays


def build_optimizers(model, cfg: Config):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and all(excluded not in name for excluded in ("embed", "head", "loop_embeddings")):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if not muon_params:
        raise RuntimeError("No Muon parameters found; check model parameter partitioning")
    muon_opt = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=cfg.adamw_lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    return muon_opt, adamw_opt


def autocast_context(device_type: str):
    if torch is None:
        raise RuntimeError("torch is required for autocast")
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def estimate_total_steps(cfg: Config, step: int, elapsed: float) -> int:
    if cfg.max_steps > 0:
        return cfg.max_steps
    if cfg.max_wallclock_seconds > 0 and step > 0 and elapsed > 0:
        return max(step + 1, int(step * cfg.max_wallclock_seconds / elapsed))
    return 1000


def maybe_save_artifact(compressed: bytes, path: str) -> None:
    if not path:
        return
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(compressed)


def quantize_tensor_int8(tensor):
    if torch is None:
        raise RuntimeError("torch is required for quantization")
    if not tensor.is_floating_point():
        return tensor.detach().cpu(), None
    max_abs = tensor.detach().abs().max()
    if max_abs.item() == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = (max_abs / 127.0).float()
    quantized = (tensor.detach() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.cpu(), scale.cpu()


def quantize_state_dict_int8(state_dict):
    quantized = {}
    for key, value in state_dict.items():
        q, scale = quantize_tensor_int8(value)
        quantized[key] = q
        if scale is not None:
            quantized[f"{key}._scale"] = scale
    return quantized


def format_bpb(loss_value: float, avg_bytes_per_token: float) -> float:
    return loss_value * LOG2_E / avg_bytes_per_token


def format_exact_bpb(loss_value: float, total_tokens: int, total_bytes: float) -> float:
    return (loss_value / math.log(2.0)) * (total_tokens / total_bytes)


def count_target_bytes(prev_ids, tgt_ids, bpb_helper: SentencePieceBPBHelper) -> float:
    if np is None:
        raise RuntimeError("numpy is required for exact tokenizer-aware val_bpb")
    prev_flat = np.asarray(prev_ids, dtype=np.int64).reshape(-1)
    tgt_flat = np.asarray(tgt_ids, dtype=np.int64).reshape(-1)
    bytes_np = bpb_helper.base_bytes_lut[tgt_flat].astype(np.int32, copy=True)
    bytes_np += (
        bpb_helper.has_leading_space_lut[tgt_flat] & ~bpb_helper.is_boundary_token_lut[prev_flat]
    ).astype(np.int32, copy=False)
    return float(bytes_np.astype(np.float64).sum())


def compute_batch_bpb(
    loss_value: float,
    inputs_np,
    targets_np,
    avg_bytes_per_token: float,
    bpb_helper: SentencePieceBPBHelper | None,
) -> float:
    if bpb_helper is None:
        return format_bpb(loss_value, avg_bytes_per_token)
    total_tokens = int(np.asarray(targets_np).size)
    total_bytes = count_target_bytes(inputs_np, targets_np, bpb_helper)
    return format_exact_bpb(loss_value, total_tokens, total_bytes)


def run_validation(
    model,
    batcher,
    val_arrays,
    batch_size: int,
    device,
    cfg: Config,
    avg_bytes_per_token: float,
    bpb_helper: SentencePieceBPBHelper | None,
) -> tuple[float, float]:
    if torch is None:
        raise RuntimeError("torch is required for validation")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    with torch.no_grad():
        if cfg.val_steps > 0:
            for _ in range(cfg.val_steps):
                batch = batcher.next_batch(batch_size, device)
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                token_count = int(targets.numel())
                total_loss += loss.item() * token_count
                total_tokens += token_count
                if bpb_helper is not None:
                    batch_np = batch.detach().cpu().numpy()
                    total_bytes += count_target_bytes(batch_np[:, :-1], batch_np[:, 1:], bpb_helper)
        else:
            if val_arrays is None:
                raise ValueError("VAL_STEPS=0 requires token arrays on disk; synthetic validation is unsupported")
            for arr in val_arrays:
                total_seqs = (len(arr) - 1) // cfg.seq_len
                for batch_seq_start in range(0, total_seqs, batch_size):
                    batch_seq_end = min(batch_seq_start + batch_size, total_seqs)
                    raw_start = batch_seq_start * cfg.seq_len
                    raw_end = batch_seq_end * cfg.seq_len + 1
                    chunk = np.asarray(arr[raw_start:raw_end], dtype=np.int64)
                    inputs_np = chunk[:-1].reshape(-1, cfg.seq_len)
                    targets_np = chunk[1:].reshape(-1, cfg.seq_len)
                    inputs = torch.from_numpy(inputs_np).to(device)
                    targets = torch.from_numpy(targets_np).to(device)
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                    token_count = int(targets.numel())
                    total_loss += loss.item() * token_count
                    total_tokens += token_count
                    if bpb_helper is not None:
                        total_bytes += count_target_bytes(inputs_np, targets_np, bpb_helper)
    if was_training:
        model.train()
    if total_tokens <= 0:
        raise ValueError("Validation did not process any tokens")
    mean_loss = total_loss / total_tokens
    if bpb_helper is None:
        return mean_loss, format_bpb(mean_loss, avg_bytes_per_token)
    return mean_loss, format_exact_bpb(mean_loss, total_tokens, total_bytes)


def main() -> int:
    cfg = Config.from_env()
    print("config:", asdict(cfg))

    if torch is None:
        print("PyTorch is not installed. Run: uv sync")
        return 1
    if np is None and (cfg.data_path or cfg.val_data_path):
        print("NumPy is required for DATA_PATH/VAL_DATA_PATH loading. Run: uv sync")
        return 1
    if cfg.tokenizer_path and spm is None:
        print("sentencepiece is required for TOKENIZER_PATH exact val_bpb support. Run: uv sync")
        return 1

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    train_bsz = max(1, cfg.train_batch_tokens // cfg.seq_len)
    val_bsz = max(1, cfg.val_batch_tokens // cfg.seq_len)
    val_path = cfg.val_data_path or cfg.data_path
    cfg.vocab_size, vocab_size_source = resolve_vocab_size(cfg, val_path)
    cfg.token_dtype, token_dtype_source = resolve_token_dtype(cfg, val_path)
    train_batcher, train_source, _ = build_batcher(cfg, cfg.data_path, cfg.seed, split="train")
    val_batcher, val_source, val_arrays = build_batcher(cfg, val_path, cfg.seed + 1, split="val")
    avg_bytes_per_token, avg_bytes_source = resolve_avg_bytes_per_token(cfg, val_path)
    bpb_helper = None
    bpb_mode = "avg_bytes_per_token"
    if cfg.tokenizer_path:
        bpb_helper = build_sentencepiece_bpb_helper(cfg.tokenizer_path, cfg.vocab_size)
        bpb_mode = "sentencepiece_exact"
    print(f"train_source={train_source} val_source={val_source} device={device}")
    print(f"vocab_size={cfg.vocab_size} source={vocab_size_source}")
    print(f"token_dtype={cfg.token_dtype} source={token_dtype_source}")
    print(f"avg_bytes_per_token={avg_bytes_per_token:.4f} source={avg_bytes_source}")
    print(f"bpb_mode={bpb_mode}")
    if cfg.tokenizer_path:
        print(f"tokenizer_path={Path(cfg.tokenizer_path).resolve()}")

    model = LoopedTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_loops=cfg.n_loops,
        max_seq_len=cfg.seq_len,
        use_smear=cfg.use_smear,
    ).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"parameters={param_count:,}")

    muon_opt, adamw_opt = build_optimizers(model, cfg)
    use_cuda_amp = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    start_time = time.perf_counter()
    qat_enabled = False
    step = 0
    last_train_loss = None
    last_val_loss = None
    best_val_loss = None
    best_val_bpb = None
    while True:
        elapsed = time.perf_counter() - start_time
        if cfg.max_steps > 0 and step >= cfg.max_steps:
            break
        if cfg.max_wallclock_seconds > 0 and elapsed >= cfg.max_wallclock_seconds:
            break

        total_steps = estimate_total_steps(cfg, step + 1, max(elapsed, 1e-6))
        qat_trigger = max(1, int(total_steps * cfg.qat_start_fraction))
        if not qat_enabled and step >= qat_trigger:
            enable_qat(model)
            qat_enabled = True
            print(f"step={step} qat=enabled")

        muon_lr = get_lr(step, total_steps, cfg.muon_lr, cfg.warmup_steps, cfg.cooldown_fraction)
        adamw_lr = get_lr(step, total_steps, cfg.adamw_lr, cfg.warmup_steps, cfg.cooldown_fraction)
        for group in muon_opt.param_groups:
            group["lr"] = muon_lr
        for group in adamw_opt.param_groups:
            group["lr"] = adamw_lr

        batch = train_batcher.next_batch(train_bsz, device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        with autocast_context(device.type):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(muon_opt)
            scaler.unscale_(adamw_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(muon_opt)
            scaler.step(adamw_opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon_opt.step()
            adamw_opt.step()
        last_train_loss = loss.item()

        if cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0:
            val_loss, val_bpb = run_validation(
                model,
                val_batcher,
                val_arrays,
                val_bsz,
                device,
                cfg,
                avg_bytes_per_token,
                bpb_helper,
            )
            last_val_loss = val_loss
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_bpb = val_bpb
            batch_np = batch.detach().cpu().numpy()
            train_bpb = compute_batch_bpb(
                loss.item(),
                batch_np[:, :-1],
                batch_np[:, 1:],
                avg_bytes_per_token,
                bpb_helper,
            )
            print(
                f"step={step} "
                f"train_loss={loss.item():.4f} "
                f"train_bpb={train_bpb:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_bpb={val_bpb:.4f} "
                f"muon_lr={muon_lr:.3e} "
                f"adamw_lr={adamw_lr:.3e} "
                f"elapsed={elapsed:.1f}s"
            )
        step += 1

    total_time = time.perf_counter() - start_time
    final_val_loss, final_val_bpb = run_validation(
        model,
        val_batcher,
        val_arrays,
        val_bsz,
        device,
        cfg,
        avg_bytes_per_token,
        bpb_helper,
    )
    if best_val_loss is None or final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_val_bpb = final_val_bpb
    quantized_state = quantize_state_dict_int8(model.state_dict())
    buffer = io.BytesIO()
    torch.save(quantized_state, buffer)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    maybe_save_artifact(compressed, cfg.artifact_path)
    code_size = Path(__file__).read_bytes()
    total_artifact = len(compressed) + len(code_size)
    summary = {
        "run_id": cfg.run_id,
        "device": str(device),
        "train_source": train_source,
        "val_source": val_source,
        "vocab_size": cfg.vocab_size,
        "token_dtype": cfg.token_dtype,
        "avg_bytes_per_token": avg_bytes_per_token,
        "bpb_mode": bpb_mode,
        "tokenizer_path": cfg.tokenizer_path or None,
        "parameters": param_count,
        "steps": step,
        "seconds": total_time,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "best_val_loss": best_val_loss,
        "best_val_bpb": best_val_bpb,
        "final_val_loss": final_val_loss,
        "final_val_bpb": final_val_bpb,
        "compressed_model_size_bytes": len(compressed),
        "code_size_bytes": len(code_size),
        "total_artifact_bytes": total_artifact,
        "artifact_budget_ok": total_artifact <= 16_000_000,
        "qat_enabled": qat_enabled,
        "config": asdict(cfg),
    }
    print("=== final_stats ===")
    print(f"steps={step}")
    print(f"seconds={total_time:.2f}")
    print(f"final_val_loss={final_val_loss:.4f}")
    print(f"final_val_bpb={final_val_bpb:.4f}")
    print(f"compressed_model_size_bytes={len(compressed)}")
    print(f"code_size_bytes={len(code_size)}")
    print(f"total_artifact_bytes={total_artifact}")
    print(f"artifact_budget_ok={total_artifact <= 16_000_000}")
    if cfg.stats_path:
        stats_path = Path(cfg.stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"stats_path={stats_path}")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
            temp_stats_path = handle.name
        print(f"stats_path={temp_stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
