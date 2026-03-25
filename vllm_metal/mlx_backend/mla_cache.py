# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx


class MLAPagedLatentCache:
    """Per-layer MLX arrays for MLA paged attention.

    Each token's cache entry is a combined latent vector [kv_norm || k_pe]:
      - kv_norm = kv_a_layernorm(compressed_kv) — the normalised KV latent
      - k_pe    = rope(k_pe_raw)                — RoPE-encoded position key

    Layout per layer: [num_blocks, block_size, latent_dim]
    where latent_dim = kv_lora_rank + qk_rope_head_dim.

    Block allocation is managed externally by the scheduler's KV cache manager.
    """

    def __init__(
        self,
        num_layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for MLA paged cache: {dtype}")

        self.num_layers = num_layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.latent_dim = kv_lora_rank + qk_rope_head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        # Per-layer latent caches: [num_blocks, block_size, latent_dim]
        self.latent_caches: list[mx.array] = [
            mx.zeros((num_blocks, block_size, self.latent_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        # Force allocation so Metal buffers exist before use
        mx.eval(*self.latent_caches)
