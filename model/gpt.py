"""Decoder-only transformer using current frontier-model building blocks.

RMSNorm, rotary position embeddings (RoPE), grouped-query attention (GQA),
SwiGLU feed-forward, weight-tied embeddings, and fused scaled-dot-product
attention. The same code scales from a laptop toy to billions of parameters by
changing ``GPTConfig``; what it cannot supply is the compute and data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    context: int = 256
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2      # GQA: fewer KV heads -> smaller KV cache
    ffn_mult: float = 8 / 3  # SwiGLU hidden size multiplier
    rope_base: float = 10_000.0
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(head_dim: int, length: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    freqs = torch.outer(torch.arange(length).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kvh = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.dim // cfg.n_heads
        self.q = nn.Linear(cfg.dim, self.h * self.hd, bias=False)
        self.kv = nn.Linear(cfg.dim, 2 * self.kvh * self.hd, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.kvh, self.hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        hidden = 64 * math.ceil(hidden / 64)
        self.w12 = nn.Linear(cfg.dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.dim), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.dim), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        cos, sin = rope_tables(cfg.dim // cfg.n_heads, cfg.context, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled residual init
            if name.endswith(("o.weight", "w3.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50,
                 should_stop: Callable[[], bool] = lambda: False):
        """Sample tokens; checks ``should_stop`` (the kill switch) every token."""
        for _ in range(max_new_tokens):
            if should_stop():
                break
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


class ByteTokenizer:
    """Byte-level tokenizer: no OOV, no training needed. Swap for BPE at scale."""
    vocab_size = 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")
