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
    parser = argparse.ArgumentParser(description="Patch PR674 train_gpt.py with PR684-style enhanced attention.")
    parser.add_argument("train_gpt", type=Path)
    args = parser.parse_args()

    path = args.train_gpt
    text = path.read_text()

    text = replace_once(
        text,
        "        self.num_heads = num_heads\n"
        "        self.num_kv_heads = num_kv_heads\n"
        "        self.head_dim = dim // num_heads\n",
        "        self.num_heads = num_heads\n"
        "        self.num_kv_heads = num_kv_heads\n"
        "        self.num_queries_per_kv = num_heads // num_kv_heads\n"
        "        self.head_dim = dim // num_heads\n",
    )

    text = replace_once(
        text,
        "        self.proj = CastedLinear(dim, dim, bias=False)\n"
        "        self.proj._zero_init = True\n"
        "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE\n",
        "        self.proj = CastedLinear(dim, dim, bias=False)\n"
        "        self.proj._zero_init = True\n"
        "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.k_gain = nn.Parameter(torch.ones(num_kv_heads, dtype=torch.float32))\n"
        "        self.k_shift_mix = nn.Parameter(torch.zeros(num_kv_heads, dtype=torch.float32))\n"
        "        self.v_shift_mix = nn.Parameter(torch.zeros(num_kv_heads, dtype=torch.float32))\n"
        "        self.local_v_mix = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))\n"
        "        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE\n",
    )

    text = replace_once(
        text,
        "        v = self.c_v(x)\n"
        "        if v_embed is not None:\n"
        "            v = v + v_embed\n"
        "        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)\n"
        "        q = F.rms_norm(q, (q.size(-1),))\n"
        "        k = F.rms_norm(k, (k.size(-1),))\n"
        "        cos, sin = self.rotary(seqlen, x.device, q.dtype)\n"
        "        q = apply_rotary_emb(q, cos, sin, self.rope_dims)\n"
        "        k = apply_rotary_emb(k, cos, sin, self.rope_dims)\n"
        "        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]\n"
        "        y = flash_attn_3_func(q, k, v, causal=True)\n"
        "        if self.use_xsa:\n"
        "            y = self._xsa_efficient(y, v)\n",
        "        v = self.c_v(x)\n"
        "        if v_embed is not None:\n"
        "            v = v + v_embed\n"
        "        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)\n"
        "        k_prev = torch.cat((torch.zeros_like(k[:, :1]), k[:, :-1]), dim=1)\n"
        "        v_prev = torch.cat((torch.zeros_like(v[:, :1]), v[:, :-1]), dim=1)\n"
        "        k = k + self.k_shift_mix.to(dtype=k.dtype)[None, None, :, None] * k_prev\n"
        "        v = v + self.v_shift_mix.to(dtype=v.dtype)[None, None, :, None] * v_prev\n"
        "        q = F.rms_norm(q, (q.size(-1),))\n"
        "        k = F.rms_norm(k, (k.size(-1),))\n"
        "        cos, sin = self.rotary(seqlen, x.device, q.dtype)\n"
        "        q = apply_rotary_emb(q, cos, sin, self.rope_dims)\n"
        "        k = apply_rotary_emb(k, cos, sin, self.rope_dims)\n"
        "        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]\n"
        "        k = k * self.k_gain.to(dtype=k.dtype)[None, None, :, None]\n"
        "        y = flash_attn_3_func(q, k, v, causal=True)\n"
        "        if self.num_heads != self.num_kv_heads:\n"
        "            v_local = v.repeat_interleave(self.num_queries_per_kv, dim=-2)\n"
        "        else:\n"
        "            v_local = v\n"
        "        y = y + self.local_v_mix.to(dtype=y.dtype)[None, None, :, None] * v_local\n"
        "        if self.use_xsa:\n"
        "            y = self._xsa_efficient(y, v)\n",
    )

    path.write_text(text)


if __name__ == "__main__":
    main()
