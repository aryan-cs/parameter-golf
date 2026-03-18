#!/usr/bin/env python3
from __future__ import annotations

import io
import math
import os
import random
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


LOG2_E = math.log2(math.e)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


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
    token_dtype: str
    vocab_size: int
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
    avg_bytes_per_token: float
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

    @classmethod
    def from_env(cls) -> "Config":
        d_model = env_int("D_MODEL", 256)
        return cls(
            run_id=os.environ.get("RUN_ID", "dev_smoke"),
            data_path=os.environ.get("DATA_PATH", ""),
            val_data_path=os.environ.get("VAL_DATA_PATH", ""),
            token_dtype=os.environ.get("TOKEN_DTYPE", "uint16"),
            vocab_size=env_int("VOCAB_SIZE", 32768),
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
            avg_bytes_per_token=env_float("AVG_BYTES_PER_TOKEN", 3.5),
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


def discover_token_arrays(data_path: str, token_dtype: str, min_length: int):
    if np is None:
        raise RuntimeError("numpy is required to load token data from disk")
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {data_path}")
    if root.is_file():
        candidates = [root]
    else:
        candidates = sorted(root.rglob("*.npy")) + sorted(root.rglob("*.bin"))
    arrays = []
    dtype = np.dtype(token_dtype)
    for candidate in candidates:
        if candidate.suffix == ".npy":
            arr = np.load(candidate, mmap_mode="r")
        else:
            arr = np.memmap(candidate, dtype=dtype, mode="r")
        arr = arr.reshape(-1)
        if arr.shape[0] > min_length:
            arrays.append(arr)
    if not arrays:
        raise FileNotFoundError(f"No token arrays longer than {min_length} found under {data_path}")
    return arrays


def build_batcher(cfg: Config, data_path: str, seed: int):
    if not data_path:
        return SyntheticTokenBatcher(cfg.vocab_size, cfg.seq_len, seed), "synthetic"
    arrays = discover_token_arrays(data_path, cfg.token_dtype, cfg.seq_len + 1)
    return PackedTokenBatcher(arrays, cfg.seq_len, seed), f"{len(arrays)} shard(s)"


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


def run_validation(model, batcher, batch_size: int, device, cfg: Config) -> float:
    if torch is None:
        raise RuntimeError("torch is required for validation")
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.val_steps):
            batch = batcher.next_batch(batch_size, device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
            losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main() -> int:
    cfg = Config.from_env()
    print("config:", asdict(cfg))

    if torch is None:
        print("PyTorch is not installed. Run: uv sync")
        return 1
    if np is None and (cfg.data_path or cfg.val_data_path):
        print("NumPy is required for DATA_PATH/VAL_DATA_PATH loading. Run: uv sync")
        return 1

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    train_bsz = max(1, cfg.train_batch_tokens // cfg.seq_len)
    val_bsz = max(1, cfg.val_batch_tokens // cfg.seq_len)
    train_batcher, train_source = build_batcher(cfg, cfg.data_path, cfg.seed)
    val_path = cfg.val_data_path or cfg.data_path
    val_batcher, val_source = build_batcher(cfg, val_path, cfg.seed + 1)
    print(f"train_source={train_source} val_source={val_source} device={device}")

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

        if step % max(1, cfg.val_loss_every) == 0:
            val_loss = run_validation(model, val_batcher, val_bsz, device, cfg)
            print(
                f"step={step} "
                f"train_loss={loss.item():.4f} "
                f"train_bpb={format_bpb(loss.item(), cfg.avg_bytes_per_token):.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_bpb={format_bpb(val_loss, cfg.avg_bytes_per_token):.4f} "
                f"muon_lr={muon_lr:.3e} "
                f"adamw_lr={adamw_lr:.3e} "
                f"elapsed={elapsed:.1f}s"
            )
        step += 1

    total_time = time.perf_counter() - start_time
    quantized_state = quantize_state_dict_int8(model.state_dict())
    buffer = io.BytesIO()
    torch.save(quantized_state, buffer)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    maybe_save_artifact(compressed, cfg.artifact_path)
    code_size = Path(__file__).read_bytes()
    total_artifact = len(compressed) + len(code_size)
    print("=== final_stats ===")
    print(f"steps={step}")
    print(f"seconds={total_time:.2f}")
    print(f"compressed_model_size_bytes={len(compressed)}")
    print(f"code_size_bytes={len(code_size)}")
    print(f"total_artifact_bytes={total_artifact}")
    print(f"artifact_budget_ok={total_artifact <= 16_000_000}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
