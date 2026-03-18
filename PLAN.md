# Parameter Golf — Winning Plan

> **Goal:** First place on the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) leaderboard.
> **Metric:** Lowest `val_bpb` (bits per byte) on the FineWeb validation set.
> **Constraints:** ≤16MB artifact (code + compressed weights), ≤10 min training on 8×H100s.
> **Deadline:** April 30, 2025.

---

## Table of Contents

1. [Understanding the Challenge](#1-understanding-the-challenge)
2. [The Metric: Bits Per Byte](#2-the-metric-bits-per-byte)
3. [The 16MB Budget](#3-the-16mb-budget)
4. [Environment Setup (Mac M4 Pro)](#4-environment-setup-mac-m4-pro)
5. [Environment Setup (RunPod H100)](#5-environment-setup-runpod-h100)
6. [Reproduce the Baseline](#6-reproduce-the-baseline)
7. [Winning Strategy Overview](#7-winning-strategy-overview)
8. [Lever 1: Tokenizer Design](#8-lever-1-tokenizer-design)
9. [Lever 2: Looped (Depth-Recurrent) Transformer Architecture](#9-lever-2-looped-depth-recurrent-transformer-architecture)
10. [Lever 3: Quantization-Aware Training (QAT)](#10-lever-3-quantization-aware-training-qat)
11. [Lever 4: The Muon Optimizer](#11-lever-4-the-muon-optimizer)
12. [Lever 5: Modern Architecture Details](#12-lever-5-modern-architecture-details)
13. [Lever 6: Test-Time Training (TTT)](#13-lever-6-test-time-training-ttt)
14. [Training Schedule and Hyperparameters](#14-training-schedule-and-hyperparameters)
15. [Ablation Workflow](#15-ablation-workflow)
16. [Submission Requirements](#16-submission-requirements)
17. [Expected bpb Progression](#17-expected-bpb-progression)
18. [Key Risks and Mitigations](#18-key-risks-and-mitigations)

---

## 1. Understanding the Challenge

The [OpenAI Parameter Golf challenge](https://openai.com/index/parameter-golf/) asks you to train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8×H100 GPUs**, evaluated by compression on the FineWeb validation set using **bits per byte (bpb)**.

The challenge is directly inspired by the NanoGPT Speedrunning challenge, but instead of optimizing for training *speed*, you are optimizing for model *quality under a size constraint* — a form of L(N) optimization.

**What makes this interesting:** The design space OpenAI explicitly wants to see explored includes:
- Test-time compute and depth recurrence
- Aggressive parameter tying
- Low-rank training
- Low precision, QAT, bitnets
- Novel tokenizers
- Test-time training, long context, megakernels

**The talent angle:** Top performers may be invited to interview for research roles at OpenAI. The challenge closes April 30.

**Helpful links:**
- GitHub repo: https://github.com/openai/parameter-golf
- OpenAI challenge page: https://openai.com/index/parameter-golf/
- OpenAI Discord: `#parameter-golf-discussions` and `#parameter-golf-announcements`
- Compute grant (RunPod credits): Apply via the form on the GitHub repo
- Participant form (optional, for recruiting): Also on the GitHub repo

---

## 2. The Metric: Bits Per Byte

Bpb is the core metric. Understanding it precisely is critical to winning.

```
bpb = cross_entropy_loss × log2(e) / avg_bytes_per_token
    = cross_entropy_loss × 1.44269 / avg_bytes_per_token
```

Where `avg_bytes_per_token` = total UTF-8 bytes in the validation text ÷ total tokens produced by your tokenizer.

**Why this is tokenizer-agnostic:** If you use a bigger vocabulary (more bytes per token), each token carries more information, so `avg_bytes_per_token` is larger, and bpb goes down even at the same raw cross-entropy loss.

**Implications:**
- A 32k-vocab tokenizer encodes ~3–4 bytes/token on English web text (vs ~1.5 for sp1024).
- This directly halves or thirds your bpb denominator — enormous free gain.
- The model sees fewer tokens per document, so context windows go further.
- Conversely, your embedding and output head are larger (32k × d_model), but with int8 quantization this is manageable.

**Validation set:** Always the fixed first 50,000 documents of `fineweb_val_*`. This never changes regardless of your tokenizer. The bpb is computed over the raw bytes of those documents.

---

## 3. The 16MB Budget

**Formula:** `artifact_size = len(train_gpt.py in bytes) + len(zlib_compress(int8_quantize(weights)))`

**Rules:**
- The cap is **decimal 16MB = 16,000,000 bytes** (not 16 MiB / 16,777,216).
- All counted code must live in `train_gpt.py` (plus any other files you include in your submission folder).
- No external downloads, no network calls, no training dataset access during evaluation.
- The artifact must be fully self-contained and reproducible.

**Budget planning:**

| Component | Estimated size |
|---|---|
| `train_gpt.py` (code) | ~50–200KB |
| Tokenizer model file (32k BPE, zlib'd) | ~500–800KB |
| Model weights (int8 + zlib) | ~13–14MB |
| **Total** | **~14–15MB** |

The baseline script prints `final_int8_zlib_roundtrip` lines at the end of training, which show you the compressed model size. Watch this carefully.

**How the baseline saves the model (from `train_gpt.py`):**
```python
# Quantize to int8
state_dict_int8 = {k: v.to(torch.int8) for k, v in model.state_dict().items()}
buf = io.BytesIO()
torch.save(state_dict_int8, buf)
compressed = zlib.compress(buf.getvalue(), level=9)
compressed_size = len(compressed)
```

Weights that have low dynamic range (e.g. from QAT) compress better. This is why QAT materially shrinks your artifact even beyond the int8 quantization step.

---

## 4. Environment Setup (Mac M4 Pro)

Your M4 Pro is the primary development machine. Use it for all architecture experiments and fast ablations. The repo ships a `train_gpt_mlx.py` that uses Apple's MLX framework.

### Step 1: Clone the repo
```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

### Step 2: Install `uv`
```bash
brew install uv
```

### Step 3: Sync the project environment
```bash
uv sync --extra mlx
```
This creates and manages the virtual environment automatically, so you do not need to run `python3 -m venv` or `source .venv/bin/activate`.

The `mlx` extra is only needed on Apple Silicon for `train_gpt_mlx.py`. On Linux or RunPod, plain `uv sync` is enough.

### Step 4: Download data (minimal subset for development)
```bash
# Downloads 1 training shard (~100M tokens) + full validation set
uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This populates:
- `./data/datasets/fineweb10B_sp1024/` — training shards (tokenized)
- `./data/tokenizers/fineweb_1024_bpe.model` — the baseline 1024-vocab SentencePiece model

### Step 5: Run the MLX smoke test
```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
uv run python train_gpt_mlx.py
```

This skips periodic validation and just prints `val_loss` and `val_bpb` once at the end. It should complete in a few minutes on M4 Pro. If this runs cleanly, your environment is working.

### Step 6: Set up for custom tokenizer experiments
```bash
# We'll train a new BPE tokenizer from scratch on the val data
# This uses the raw text (not pre-tokenized), so we need the HuggingFace dataset
uv run python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
# We only need a sample to train the tokenizer
"
```

---

## 5. Environment Setup (RunPod H100)

Use RunPod for 1×H100 experiments (hyperparameter tuning) and final 8×H100 submission runs.

### Step 1: Create a RunPod account and add SSH key
Go to https://runpod.io → Settings → SSH Keys → add your public key (`~/.ssh/id_ed25519.pub`).

### Step 2: Launch a pod
- For experiments: **1×H100 SXM** (~$2.50–3.50/hr)
- For final submission: **8×H100 SXM** (~$20/hr) — use sparingly

Select the **RunPod PyTorch** template (comes with CUDA, PyTorch, Python pre-installed).

### Step 3: SSH into the pod
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### Step 4: Clone repo and install dependencies
```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Step 5: Download data on the pod
```bash
# For experiments: 1 shard is enough
uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# For final submission run: download more shards (each is ~100M tokens)
uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20
```

### Step 6: Run baseline on 1×H100
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Expected output at the end:
```
val_loss: ~2.0   val_bpb: ~1.20   compressed_size: <16MB
```

### Step 7: Apply for compute grant
Fill out the form linked from the GitHub repo to get RunPod credits. Do this immediately — $1M in credits are available while supplies last.

---

## 6. Reproduce the Baseline

Before changing anything, get the baseline working end-to-end. This validates your environment and gives you a reference bpb to beat.

**Full baseline command (CUDA, single GPU):**
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Useful environment variables:**
```bash
MAX_WALLCLOCK_SECONDS=0      # Remove the 10-minute cap for local experiments
VAL_LOSS_EVERY=200           # Print validation metrics every 200 steps
VAL_BATCH_SIZE=8192          # Tokens to use for validation
```

**What to record from this run:**
- `val_bpb` (your baseline: ~1.20)
- `compressed_size` (must stay under 16,000,000 bytes)
- Training tokens per second (your throughput benchmark)

---

## 7. Winning Strategy Overview

bpb improvement comes from six independent, compounding levers. Apply them in order.

```
bpb = (cross_entropy_loss × 1.44269) / avg_bytes_per_token
       ^                                ^
       Improved by: architecture,        Improved by: better tokenizer
       optimizer, QAT, TTT
```

| Lever | What it does | Expected bpb improvement | Effort |
|---|---|---|---|
| 1. Tokenizer (32k vocab) | More bytes/token → better denominator | ~15–20% | Low |
| 2. Looped transformer | More effective depth, same params | ~10–20% | Medium |
| 3. QAT (int8) | Weights compress better → more params fit | ~5–10% | Medium |
| 4. Muon optimizer | ~2× more effective gradient updates | ~10–15% | Low |
| 5. Modern arch details | RoPE, RMSNorm, QK-Norm, SwiGLU | ~5–10% | Low |
| 6. Test-time training | Gradient steps on val docs at eval | ~10–20% | High |

These stack multiplicatively. Getting levers 1–5 right before attempting lever 6 is the right order.

---

## 8. Lever 1: Tokenizer Design

This is the single highest-leverage change you can make with the least model quality work. A bigger vocabulary means more bytes per token, which directly improves bpb at the same raw cross-entropy loss.

### Why the baseline sp1024 is weak

With only 1024 tokens, each token encodes very few bytes (~1.3 bytes/token on English text). This means your model has to do enormous work per byte, and bpb is penalized by the tiny denominator.

### Design a 32k BPE tokenizer

```python
# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import os

# Stream a sample of FineWeb training data for tokenizer training
# (we don't need much — 1–2GB of text is enough for a good BPE vocab)
def get_training_text(n_docs=50000):
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    texts = []
    for i, doc in enumerate(ds):
        if i >= n_docs:
            break
        texts.append(doc["text"])
    return texts

texts = get_training_text(50000)

# Train a ByteLevel BPE tokenizer (handles any Unicode, no UNK tokens)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    texts,
    vocab_size=32768,          # 32k vocab — good balance of bytes/token vs embed size
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)

# Save
os.makedirs("./data/tokenizers", exist_ok=True)
tokenizer.save_model("./data/tokenizers", "fineweb_32k_bpe")
print("Saved tokenizer")
```

### Measure the tokenizer file size (it counts toward 16MB)
```python
import os, zlib

with open("./data/tokenizers/fineweb_32k_bpe-vocab.json", "rb") as f:
    vocab_bytes = f.read()
with open("./data/tokenizers/fineweb_32k_bpe-merges.txt", "rb") as f:
    merges_bytes = f.read()

combined = vocab_bytes + merges_bytes
compressed = zlib.compress(combined, level=9)
print(f"Tokenizer raw size: {len(combined)/1e6:.2f}MB")
print(f"Tokenizer compressed: {len(compressed)/1e6:.2f}MB")
# Target: <1MB compressed
```

### Measure bytes per token gain
```python
import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer

# Load both tokenizers
sp = spm.SentencePieceProcessor()
sp.load("./data/tokenizers/fineweb_1024_bpe.model")

bpe32k = ByteLevelBPETokenizer(
    "./data/tokenizers/fineweb_32k_bpe-vocab.json",
    "./data/tokenizers/fineweb_32k_bpe-merges.txt",
)

sample_text = "This is a sample of English web text to measure tokenization efficiency."
sp_tokens = sp.encode(sample_text)
bpe_tokens = bpe32k.encode(sample_text).ids

print(f"sp1024:  {len(sp_tokens)} tokens, {len(sample_text.encode())/len(sp_tokens):.2f} bytes/token")
print(f"bpe32k:  {len(bpe_tokens)} tokens, {len(sample_text.encode())/len(bpe_tokens):.2f} bytes/token")
# Expect ~1.3 vs ~3.5+ bytes/token
```

### Re-tokenize the dataset with your new tokenizer

The challenge's data pipeline tokenizes training data ahead of time. You'll need to re-export with your new tokenizer. See `data/README.md` in the repo for the full export pipeline. The key environment variables are:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_export_root \
uv run python data/cached_challenge_fineweb.py --variant your_32k_variant --train-shards 20
```

Alternatively for fast iteration: tokenize on-the-fly in your `train_gpt.py` using a streaming approach. This is slower but avoids the pre-tokenization step.

---

## 9. Lever 2: Looped (Depth-Recurrent) Transformer Architecture

This is the architectural core of the winning submission. A looped transformer reuses the same set of parameters N times, giving you effective depth of N×L at the cost of L layers.

### Core idea

```
Standard 24-layer transformer:
  Layer 1 weights → Layer 2 weights → ... → Layer 24 weights
  (24 independent weight sets, 24× the parameters)

Looped 6-layer transformer, N=4 loops:
  [Layer 1..6 weights] → [same Layer 1..6 weights] → [same] → [same]
  (1 weight set, applied 4 times = effective 24-layer depth)
```

This is far more parameter-efficient. The weights compress extremely well (only one copy exists in the checkpoint, and the repeated application is handled in the forward pass code, not the weights).

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SharedTransformerBlock(nn.Module):
    """A single transformer block whose weights are reused N times."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Pre-norm (RMSNorm is more stable and parameter-free scale)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        # Attention — fused QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-Norm: normalizes Q and K before attention — prevents attention entropy collapse
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

        # FFN with SwiGLU activation (gated, more expressive than ReLU)
        # d_ff is typically 8/3 * d_model for SwiGLU to match param count of 4*d_model ReLU
        self.ff_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ff_up   = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def attention(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        q, k = q.transpose(1,2), k.transpose(1,2)  # (B, H, T, D)
        v = v.transpose(1,2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q, k = apply_rotary(q, k, cos, sin)

        # Flash attention (PyTorch ≥2.0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        x = x.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(x)

    def ffn(self, x):
        # SwiGLU: gate × silu(up) → down
        return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))

    def forward(self, x, cos, sin, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_loops, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_loops = n_loops

        # Embedding (NOT tied to output head — allows independent optimization)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        # THE KEY: a single shared block, applied n_loops times
        self.block = SharedTransformerBlock(d_model, n_heads, d_ff)

        # Optional: per-loop embeddings so the model knows which iteration it's on
        # These are tiny (n_loops × d_model) and make a measurable difference
        self.loop_embeddings = nn.Embedding(n_loops, d_model)

        # Final norm and output head
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for name, p in self.block.named_parameters():
            if 'weight' in name and p.dim() == 2:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.n_loops))

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids)  # (B, T, d_model)
        cos, sin = self.rope(T)

        # Apply the shared block n_loops times
        for loop_idx in range(self.n_loops):
            # Inject loop position embedding
            loop_emb = self.loop_embeddings(
                torch.tensor(loop_idx, device=x.device)
            ).unsqueeze(0).unsqueeze(0)
            x = x + loop_emb
            x = self.block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits


# Example configuration — tune these:
model = LoopedTransformer(
    vocab_size=32768,   # Your 32k BPE vocab
    d_model=512,        # Hidden dimension
    n_heads=8,          # Attention heads (d_head = 512/8 = 64)
    d_ff=1365,          # SwiGLU FFN dim (≈ 8/3 * d_model to match 4*d_model ReLU params)
    n_loops=8,          # Repeat the block 8 times
    max_seq_len=1024,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# With the above config: ~15–20M params
# At int8 (1 byte/param) + zlib: ~8–12MB
```

### Choosing the right configuration

The key tradeoff: `d_model` × `n_loops` × compute vs. compressed size.

Start with these ablation points:

| d_model | n_heads | d_ff | n_loops | ~Params | ~Compressed |
|---|---|---|---|---|---|
| 384 | 6 | 1024 | 6 | 8M | 5MB |
| 512 | 8 | 1365 | 8 | 18M | 10MB |
| 640 | 8 | 1706 | 10 | 28M | 14MB |
| 768 | 12 | 2048 | 8 | 50M | **too big** |

The sweet spot is likely around 512–640 d_model with 8–10 loops.

---

## 10. Lever 3: Quantization-Aware Training (QAT)

The challenge quantizes your weights to int8 and compresses with zlib before counting size. Training with QAT means the model *expects* to be quantized — so the final int8 checkpoint is far more accurate than post-training quantization.

### Strategy: Two-phase training

**Phase 1 (first 60% of steps): Standard bf16 training**
The model learns in full precision. Do not quantize yet — QAT from step 0 can destabilize training.

**Phase 2 (final 40% of steps): Fake int8 quantization**
Insert fake quantization into the forward pass. The model learns to be robust to int8 rounding.

### Implementation

```python
def fake_quantize_int8(tensor):
    """Simulate int8 quantization during forward pass."""
    # Per-tensor symmetric int8: scale based on max absolute value
    scale = tensor.abs().max() / 127.0
    scale = scale.clamp(min=1e-8)
    # Quantize and dequantize
    quantized = (tensor / scale).round().clamp(-128, 127)
    return quantized * scale  # Dequantized — same dtype as input, but values snapped to int8 grid

class QATLinear(nn.Module):
    """Linear layer with optional QAT applied to weights."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.qat_enabled = False

    def enable_qat(self):
        self.qat_enabled = True

    def forward(self, x):
        weight = fake_quantize_int8(self.linear.weight) if self.qat_enabled else self.linear.weight
        return F.linear(x, weight, self.linear.bias)

# In your training loop:
def enable_qat(model):
    for module in model.modules():
        if isinstance(module, QATLinear):
            module.enable_qat()
    print("QAT enabled")

# Usage:
step = 0
total_steps = 10000
qat_start_step = int(total_steps * 0.6)

for batch in dataloader:
    if step == qat_start_step:
        enable_qat(model)
    # ... normal training step ...
    step += 1
```

### Why int8 and not 1-bit (BitNet)?

At the small scale of this challenge (~10–30M params), 1-bit quantization loses too much model capacity. Int8 compresses 2× vs bf16 and has minimal accuracy degradation, making it the right choice. You can experiment with 4-bit if you want to push further, but int8 is the safe default.

---

## 11. Lever 4: The Muon Optimizer

Muon replaces AdamW for all 2D weight matrices (the bulk of your model). It is consistently 1.5–2× more compute-efficient than AdamW on language model training tasks, meaning you train more effectively in the same 10-minute window.

**Critical rule:** Muon only applies to 2D parameters. Embeddings, layer norms, biases, and 1D parameters must still use AdamW.

### Implementation

```python
import torch
import torch.nn.functional as F

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute G / sqrt(G^T G).
    This is the core of Muon — it orthogonalizes the gradient matrix.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon — orthogonalized gradient descent for 2D weight matrices.
    Use alongside AdamW for 1D params (embeddings, norms, biases).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    continue  # Skip 1D params — use AdamW for those
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g + momentum * buf
                else:
                    g = buf
                # Orthogonalize the gradient update
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                # Scale update to match RMS of parameters
                g *= max(1, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


def build_optimizers(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.1):
    """
    Split model parameters: Muon for 2D matrices, AdamW for everything else.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Muon: 2D weight matrices (linear layers in the shared block)
        # Exclude embeddings and output head — these need AdamW
        if (p.ndim == 2
                and 'embed' not in name
                and 'head' not in name
                and 'loop_embeddings' not in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=0.95)
    adamw_opt = torch.optim.AdamW(
        adamw_params, lr=adamw_lr,
        betas=(0.9, 0.95), weight_decay=weight_decay
    )
    return muon_opt, adamw_opt
```

### Learning rate schedule

Use a warmup + cosine decay + long linear cooldown:

```python
def get_lr(step, total_steps, max_lr, warmup_steps=200, cooldown_fraction=0.3):
    """
    Warmup → cosine peak → linear cooldown.
    The long cooldown is important — don't cut it short.
    """
    cooldown_start = int(total_steps * (1 - cooldown_fraction))

    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    elif step < cooldown_start:
        # Cosine decay from peak to 10% of peak
        progress = (step - warmup_steps) / (cooldown_start - warmup_steps)
        return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    else:
        # Linear decay to near zero
        progress = (step - cooldown_start) / (total_steps - cooldown_start)
        return max_lr * 0.1 * (1 - progress)
```

---

## 12. Lever 5: Modern Architecture Details

Small details that compound:

### RoPE (Rotary Positional Embeddings)
Already included in the implementation above. RoPE has no learnable parameters (unlike learned positional embeddings), saving budget for model capacity. It also extrapolates better to longer sequences.

### RMSNorm (Root Mean Square Layer Norm)
```python
# nn.RMSNorm is built into PyTorch ≥2.1
# It's equivalent to LayerNorm without the mean-centering — faster and works as well
self.norm = nn.RMSNorm(d_model)
```

### QK-Norm (Query-Key Normalization)
Normalizes Q and K vectors independently before computing attention scores. Prevents attention entropy collapse (where one head attends to everything or nothing). Already included in the implementation above. Critical for training stability in looped architectures.

### SwiGLU Activation
```python
# SwiGLU: out = down(silu(gate(x)) * up(x))
# More expressive than ReLU for the same parameter count.
# Use d_ff = int(8/3 * d_model) to match a standard 4*d_model ReLU FFN in total params.
```

### The "Smear" Operation (from modded-nanogpt speedrun)
A cheap operation that lets each token peek one position backward:
```python
class SmearBlock(nn.Module):
    """
    Learned weighted sum of current token and previous token.
    Very cheap (1 param per dim) but measurably helps.
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Shift x by 1 position
        x_prev = torch.roll(x, 1, dims=1)
        x_prev[:, 0, :] = 0  # No look-back at position 0
        return x + torch.sigmoid(self.gate) * x_prev
```

Add this as a final step after the embedding, before the main loop.

### Gradient Clipping
Essential for training stability in looped architectures:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 13. Lever 6: Test-Time Training (TTT)

This is the highest-ceiling technique. Before evaluating loss on each validation document, run a few gradient update steps *on that document itself*. The model briefly adapts to the document's style and distribution before generating predictions.

**Is this allowed?** The rules say no external data and no network calls during evaluation. The validation data itself is already downloaded locally. This should be permitted — but verify on the Discord (`#parameter-golf-discussions`) before your final submission.

### Implementation

```python
@torch.enable_grad()
def test_time_train(model, tokens, n_steps=3, lr=1e-4):
    """
    Run n_steps of gradient updates on a single document before evaluating loss.
    Uses a fresh optimizer state each call — only updates for this document.
    """
    model.train()

    # Only update the shared block weights, not embed/head
    # (preserves general language knowledge while adapting to document style)
    ttt_params = [p for name, p in model.named_parameters()
                  if 'block' in name and p.requires_grad]

    optimizer = torch.optim.Adam(ttt_params, lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate_with_ttt(model, val_dataset, n_ttt_steps=3, ttt_lr=1e-4):
    """
    Evaluate bpb on the validation set with test-time training.
    """
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for doc_tokens, doc_bytes in val_dataset:
        # Save a checkpoint of the model state
        original_state = {k: v.clone() for k, v in model.state_dict().items()
                          if 'block' in k}

        # Adapt to this document
        model = test_time_train(model, doc_tokens, n_steps=n_ttt_steps, lr=ttt_lr)

        # Evaluate on this document
        with torch.no_grad():
            model.eval()
            logits = model(doc_tokens[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                doc_tokens[:, 1:].reshape(-1)
            )
            total_loss += loss.item() * doc_tokens.numel()
            total_tokens += doc_tokens.numel()
            total_bytes += doc_bytes

        # Restore original weights (so we don't pollute across documents)
        for k, v in original_state.items():
            model.state_dict()[k].copy_(v)

    avg_loss = total_loss / total_tokens
    bpb = avg_loss * math.log2(math.e) / (total_bytes / total_tokens)
    return bpb
```

**TTT hyperparameters to tune:**
- `n_ttt_steps`: 1–10 gradient steps. More steps = better adaptation but slower eval.
- `ttt_lr`: 1e-5 to 1e-3. Too high = catastrophic forgetting.
- Which params to update: updating only the FFN weights tends to be more stable than updating attention.

---

## 14. Training Schedule and Hyperparameters

### Full training loop skeleton

```python
import os, io, math, zlib, time
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
import torch.distributed as dist

# --- Config (set via environment variables for easy sweeping) ---
VOCAB_SIZE     = int(os.environ.get("VOCAB_SIZE", 32768))
D_MODEL        = int(os.environ.get("D_MODEL", 512))
N_HEADS        = int(os.environ.get("N_HEADS", 8))
N_LOOPS        = int(os.environ.get("N_LOOPS", 8))
D_FF           = int(os.environ.get("D_FF", 1365))
SEQ_LEN        = int(os.environ.get("SEQ_LEN", 1024))
BATCH_TOKENS   = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))  # ~512k tokens/step
MUON_LR        = float(os.environ.get("MUON_LR", 0.02))
ADAMW_LR       = float(os.environ.get("ADAMW_LR", 3e-4))
MAX_WALLCLOCK  = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 590))   # 10min - 10s buffer
QAT_FRACTION   = float(os.environ.get("QAT_FRACTION", 0.4))          # Last 40% of steps use QAT
WARMUP_STEPS   = int(os.environ.get("WARMUP_STEPS", 200))

# --- DDP setup ---
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True

# --- Build model ---
model = LoopedTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LOOPS, SEQ_LEN).to(device)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_rank])
raw_model = model.module if ddp else model

# --- Optimizers ---
muon_opt, adamw_opt = build_optimizers(
    raw_model, muon_lr=MUON_LR, adamw_lr=ADAMW_LR
)

# --- Training loop ---
start_time = time.time()
step = 0
total_steps_estimate = 2000  # will update dynamically

scaler = torch.cuda.amp.GradScaler()

while True:
    elapsed = time.time() - start_time
    if elapsed >= MAX_WALLCLOCK:
        break

    # Dynamic step estimate (for LR schedule and QAT trigger)
    remaining = MAX_WALLCLOCK - elapsed
    # Estimate total steps based on current throughput

    # Enable QAT at 60% of estimated total steps
    qat_trigger = int(total_steps_estimate * (1 - QAT_FRACTION))
    if step == qat_trigger:
        enable_qat(raw_model)
        if master_process:
            print(f"Step {step}: QAT enabled")

    # Get learning rates
    muon_lr_now = get_lr(step, total_steps_estimate, MUON_LR)
    adamw_lr_now = get_lr(step, total_steps_estimate, ADAMW_LR)
    for g in muon_opt.param_groups:
        g['lr'] = muon_lr_now
    for g in adamw_opt.param_groups:
        g['lr'] = adamw_lr_now

    # Forward pass with automatic mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        tokens = next(train_iter)  # (B, T+1)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            tokens[:, 1:].reshape(-1)
        )

    # Backward
    muon_opt.zero_grad()
    adamw_opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(muon_opt)
    scaler.unscale_(adamw_opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(muon_opt)
    scaler.step(adamw_opt)
    scaler.update()

    if master_process and step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f} lr_muon={muon_lr_now:.2e} elapsed={elapsed:.0f}s")

    step += 1

# --- Final: quantize, compress, measure ---
if master_process:
    state_dict = raw_model.state_dict()
    state_dict_int8 = {k: v.to(torch.int8) for k, v in state_dict.items()}
    buf = io.BytesIO()
    torch.save(state_dict_int8, buf)
    compressed = zlib.compress(buf.getvalue(), level=9)
    code_bytes = len(open(__file__, 'rb').read())

    print(f"\n=== Final Stats ===")
    print(f"Model compressed size: {len(compressed)/1e6:.3f}MB")
    print(f"Code size: {code_bytes/1e6:.3f}MB")
    print(f"Total artifact: {(len(compressed) + code_bytes)/1e6:.3f}MB / 16.000MB")
    assert len(compressed) + code_bytes <= 16_000_000, "OVER BUDGET!"
```

### Recommended starting hyperparameters

```bash
# On 1×H100 for experiments:
RUN_ID=loop_qat_muon_32k \
VOCAB_SIZE=32768 \
D_MODEL=512 \
N_HEADS=8 \
N_LOOPS=8 \
D_FF=1365 \
MUON_LR=0.02 \
ADAMW_LR=3e-4 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=0 \  # No cap for experiments
uv run torchrun --standalone --nproc_per_node=1 train_gpt.py

# On 8×H100 for final submission:
RUN_ID=final_submission \
VOCAB_SIZE=32768 \
D_MODEL=640 \
N_HEADS=8 \
N_LOOPS=10 \
D_FF=1706 \
MUON_LR=0.02 \
ADAMW_LR=3e-4 \
TRAIN_BATCH_TOKENS=2097152 \  # Scale batch with GPU count
MAX_WALLCLOCK_SECONDS=590 \
uv run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## 15. Ablation Workflow

Fast, disciplined iteration is how you win. Run ablations in this order:

### Step 1: Baseline bpb (your reference)
```
baseline (sp1024, standard GPT): ~1.20 bpb
```

### Step 2: Tokenizer swap only (control everything else)
```
+ 32k BPE tokenizer, same architecture: target ~1.00–1.05 bpb
```

### Step 3: Add looped architecture (keep 32k tokenizer)
```
+ 6-layer block × 8 loops: target ~0.90–0.95 bpb
```

### Step 4: Add Muon optimizer
```
+ Muon for 2D params: target ~0.85–0.90 bpb
```

### Step 5: Add QAT
```
+ int8 QAT from 60% of training: verify budget still <16MB
```

### Step 6: Tune loop count and d_model
```
Sweep: N_LOOPS ∈ {6, 8, 10, 12}, D_MODEL ∈ {384, 512, 640}
Keep compressed size <15MB (leave 1MB buffer for code)
```

### Step 7: Test-time training
```
+ TTT with 3–5 gradient steps per val doc: target <0.80 bpb
```

### Tracking ablations

Keep a simple log. For each run, record:
- `val_bpb`
- `compressed_model_size_mb`
- `total_artifact_mb`
- `training_steps`
- `elapsed_seconds`
- Config: `d_model`, `n_loops`, `vocab_size`, `muon_lr`, `qat_start_fraction`

---

## 16. Submission Requirements

Once you have a strong result, submit a PR to the repo. All submissions go in `/records/`.

### Required files in your submission folder

```
records/
  your_submission_name/
    README.md          ← Required: explain your approach in detail
    submission.json    ← Required: structured metadata
    train_log.txt      ← Required: auto-produced by your script
    train_gpt.py       ← Required: must compile and run from this folder
```

### `submission.json` format (follow the example runs in the repo)

```json
{
  "name": "Your Name",
  "github_id": "your_github_username",
  "val_bpb": 0.847,
  "compressed_model_size_bytes": 13500000,
  "code_size_bytes": 45000,
  "total_artifact_bytes": 13545000,
  "training_steps": 1842,
  "training_seconds": 587,
  "gpus": "8xH100 SXM",
  "vocab_size": 32768,
  "d_model": 640,
  "n_loops": 10,
  "optimizer": "Muon + AdamW",
  "qat": true,
  "test_time_training": false,
  "notes": "Looped transformer with 32k BPE tokenizer, Muon optimizer, int8 QAT"
}
```

### `README.md` — what to include

1. **Architecture overview:** What makes your model different from the baseline
2. **Key techniques:** Tokenizer choice, architecture details, optimizer, QAT setup
3. **Ablation results:** bpb at each step of your development
4. **Training command:** The exact `uv run torchrun` command to reproduce your result
5. **Results table:** `val_bpb`, `compressed_size`, `training_time`
6. **What you tried that didn't work** (optional, but valued)

### Submission rules (from the repo)

- PRs must only add a new folder to `/records/` — no edits to other files
- The `train_gpt.py` must compile and run successfully from within the records folder
- Broken scripts are rejected
- The script must run in under 10 minutes on 8×H100s (for leaderboard submissions)
- No network calls, external downloads, or training data access during evaluation
- Non-record submissions (interesting approaches that don't beat SOTA) are also welcome — just note it

### After submitting

- Join the OpenAI Discord: `#parameter-golf-discussions`
- Fill out the Challenge Participant Form (linked from the repo) — this is how OpenAI attributes submissions and reaches out about job opportunities
- Apply for the compute grant if you haven't already

---

## 17. Expected bpb Progression

| Stage | What changes | Expected val_bpb |
|---|---|---|
| Baseline (sp1024, standard GPT) | Nothing | ~1.20 |
| + 32k BPE tokenizer | Better bytes/token | ~1.00–1.05 |
| + Looped architecture (8 loops) | More effective depth | ~0.90–0.95 |
| + Muon optimizer | 2× training efficiency | ~0.85–0.90 |
| + Modern arch details (QK-Norm, SwiGLU, RoPE) | Better training stability | ~0.82–0.87 |
| + int8 QAT (last 40% of training) | Allows more params in budget | ~0.80–0.85 |
| + Test-time training (3–5 steps/doc) | Adapts to each val doc | ~0.70–0.80 |

These are estimates based on literature results and expected leverage. The actual improvements depend on careful hyperparameter tuning. Your job is to realize as much of this as possible.

---

## 18. Key Risks and Mitigations

### Risk: Looped transformer is unstable
**Symptoms:** Loss explodes or doesn't decrease after a few hundred steps.
**Mitigations:**
- Use QK-Norm (already in the implementation) — this is the single most important stabilizer for looped architectures
- Use per-loop embeddings so the model has a sense of depth position
- Reduce initial learning rate and use longer warmup (400 steps instead of 200)
- Clip gradients aggressively (max_norm=0.5 instead of 1.0)
- Use pre-norm (norm before attention/FFN), not post-norm

### Risk: 32k tokenizer exceeds 16MB budget
**Diagnosis:** Check `len(zlib.compress(tokenizer_bytes, 9))`. A well-trained 32k BPE on English text typically compresses to 500–800KB.
**Mitigations:**
- Use ByteLevel BPE (HuggingFace `tokenizers`) rather than SentencePiece — the vocab.json compresses better
- If still too large, reduce to 16k vocab (~300KB compressed) — still a massive improvement over sp1024

### Risk: 8×H100 run exceeds 10 minutes
**Diagnosis:** Profile step time on 1×H100 and multiply by 8. The looped architecture serializes each loop iteration — this does not parallelize across loops, only across the batch dimension (DDP).
**Mitigations:**
- Reduce `N_LOOPS` for the final submission (8 → 6) and compensate by increasing `D_MODEL`
- Use `torch.compile()` — significant speedup on H100s
- Ensure your data loader is not the bottleneck (prefetch with `pin_memory=True`, `num_workers=4`)

### Risk: QAT degrades model quality
**Diagnosis:** val_bpb worsens when QAT is enabled.
**Mitigations:**
- Start QAT later (70% of steps instead of 60%)
- Lower the Muon learning rate by 2× when QAT starts
- Use per-channel int8 quantization instead of per-tensor (more accurate, slightly more complex)

### Risk: Test-time training causes catastrophic forgetting across documents
**Symptoms:** val_bpb on early documents is great but worsens on later ones.
**Mitigations:**
- Always restore model weights to the pre-TTT checkpoint after each document (the implementation above already does this)
- Only update FFN weights during TTT, not attention (FFN is more document-style-specific)
- Use a very low TTT learning rate (1e-5)

### Risk: Code is too large (eats into model weight budget)
**Diagnosis:** `len(open('train_gpt.py', 'rb').read())` > 500KB
**Mitigations:**
- Remove all comments and docstrings before final submission
- Use short variable names in the submission version (not during development)
- Muon implementation is ~40 lines — keep it lean

---

*Good luck. First place is achievable. Ship fast, iterate constantly, measure everything.*
