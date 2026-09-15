"""Mock vLLM-style paged KV cache generator (trimmed from
benchmark/kvstore/kvstore_benchmark.py, bf16 only).

Layout: [2 (K/V), num_blocks, block_tokens, kv_heads, head_dim], contiguous, so
(kv, block) pairs are contiguous chunks of `block_bytes`.
"""

import math

import torch

DEFAULT_SHAPE = (2, 1024, 16, 4, 128)  # 16 KiB per K or V block, 32 MiB total
SEED = 20240517


def generate_kv_cache(shape=DEFAULT_SHAPE, device="cuda", seed=SEED) -> torch.Tensor:
    kv_count, num_blocks, block_tokens, kv_heads, head_dim = shape
    latent = min(32, max(8, head_dim // 8))
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*s):
        return torch.randn(s, device=device, generator=g)

    block_ctx = rnd(num_blocks, 1, kv_heads, latent)
    token_innov = rnd(num_blocks, block_tokens, kv_heads, latent)
    block_drift = rnd(num_blocks, 1, kv_heads, latent)
    pos = torch.linspace(-1.0, 1.0, block_tokens, device=device).view(1, block_tokens, 1, 1)
    hidden = 0.65 * block_ctx + 0.70 * token_innov + 0.15 * pos * block_drift

    proj = rnd(kv_count, kv_heads, latent, head_dim) / math.sqrt(latent)
    kv = torch.einsum("bthr,khrd->kbthd", hidden, proj)
    kv.mul_(rnd(kv_count, 1, 1, kv_heads, head_dim).mul_(0.25).exp_())
    return kv.to(torch.bfloat16).contiguous()


def block_bytes(shape, dtype=torch.bfloat16) -> int:
    return math.prod(shape[2:]) * torch.tensor([], dtype=dtype).element_size()
