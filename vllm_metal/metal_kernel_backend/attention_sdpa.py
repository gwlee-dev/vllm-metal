# SPDX-License-Identifier: Apache-2.0
"""Scaled dot-product attention (SDPA) via MLX native flash attention.

Uses ``mx.fast.scaled_dot_product_attention`` (flash attention with tiled
online softmax) for both prefill and decode.  Paged KV cache is gathered
into contiguous tensors per request before the SDPA call — this trades a
cheap memory copy for MLX's highly optimized fused attention kernel, which
is 2-11x faster than the custom Metal paged kernel at all context lengths.

Supports MHA, GQA, and MQA — MLX SDPA handles head broadcasting natively.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` linear projections
- ``rope`` for rotary position embeddings
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm`` (Qwen3 per-head RMSNorm before RoPE)

Covers: Qwen3, Llama, Mistral, and other standard transformer architectures.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.packed_prefill_compat import (
    apply_packed_rope,
)
from vllm_metal.paged_attention_common import PagedAttentionContext

# === Metal kernel block-size support ===
# The paged attention Metal kernel is template-instantiated for these block
# sizes only.  Sorted descending so _pick_kernel_block_size selects the
# largest valid divisor first, minimising the block-table expansion ratio.
_KERNEL_BLOCK_SIZES = (32, 16, 8)


def is_sdpa(module: nn.Module) -> bool:
    """Return True if *module* is an SDPA attention layer (MHA, GQA, or MQA)."""
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


# === Block-size translation helpers ===


def _pick_kernel_block_size(cache_block_size: int) -> int:
    """Pick the largest kernel-supported block size that divides evenly."""
    for kbs in _KERNEL_BLOCK_SIZES:
        if cache_block_size % kbs == 0:
            return kbs
    raise ValueError(
        f"Cache block_size={cache_block_size} is not divisible by any "
        f"supported kernel block size {_KERNEL_BLOCK_SIZES}. "
        "Adjust VLLM_METAL_BLOCK_SIZE or the hybrid page alignment."
    )


def _build_block_tables(
    raw_block_tables: list[list[int]],
    cache_block_size: int,
) -> tuple[mx.array, int]:
    """Build kernel-compatible block tables, translating if necessary.

    When ``cache_block_size`` exceeds the kernel's compiled block sizes,
    each vLLM block ``b`` is expanded into ``ratio`` kernel blocks
    ``[b*ratio, b*ratio+ratio)``.  The cache is reshaped later to
    match (zero-copy).

    Returns:
        (block_tables, kernel_block_size)
    """
    if not raw_block_tables:
        return mx.zeros((0, 0), dtype=mx.int32), cache_block_size

    if cache_block_size in _KERNEL_BLOCK_SIZES:
        # Fast path — no translation needed.
        max_blocks = max(len(bt) for bt in raw_block_tables)
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
        return mx.array(padded, dtype=mx.int32), cache_block_size

    # Hybrid path — translate large block_size to a kernel-compatible one.
    # Vectorized: each vLLM block b → [b*ratio, b*ratio+1, …, b*ratio+ratio-1].
    kernel_bs = _pick_kernel_block_size(cache_block_size)
    ratio = cache_block_size // kernel_bs

    max_blocks = max(len(bt) for bt in raw_block_tables)
    padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
    bt_arr = mx.array(padded, dtype=mx.int32)  # [num_seqs, max_blocks]
    offsets = mx.arange(ratio, dtype=mx.int32)  # [ratio]
    # [num_seqs, max_blocks, 1] * ratio + [1, 1, ratio] → [num_seqs, max_blocks, ratio]
    expanded = (bt_arr[:, :, None] * ratio + offsets[None, None, :]).reshape(
        bt_arr.shape[0], -1
    )
    return expanded, kernel_bs


# === Causal mask helper ===


def _make_causal_mask(
    num_new: int, ctx_len: int, past_len: int, dtype: mx.Dtype
) -> mx.array:
    """Build an additive causal mask for continuation chunks.

    New tokens (past_len .. past_len+num_new-1) attend to all past tokens
    (0 .. past_len-1) plus causally among themselves.

    Returns: [1, 1, num_new, ctx_len] additive mask (0.0 = attend, -inf = block).
    """
    # Vectorized: new token i attends to KV positions 0..past_len+i
    rows = mx.arange(num_new)[:, None] + past_len  # [num_new, 1]
    cols = mx.arange(ctx_len)[None, :]  # [1, ctx_len]
    mask = mx.where(cols <= rows, 0.0, mx.finfo(dtype).min).astype(dtype)
    return mask[None, None, :, :]  # [1, 1, num_new, ctx_len]


# === SDPA forward ===


def sdpa_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
) -> mx.array:
    """Full SDPA forward pass: project → norm → RoPE → Metal kernel.

    Handles MHA, GQA, and MQA uniformly — the head ratio between
    query and KV heads is passed to the Metal kernel which handles
    the broadcast internally.
    """
    B, L, D = x.shape  # noqa: N806

    # Resolve head counts — mlx_lm uses different attribute names:
    #   Qwen3/Llama/Gemma: n_heads, n_kv_heads
    #   Qwen3.5 (Qwen3Next): num_attention_heads, num_key_value_heads
    n_heads = getattr(inner, "n_heads", None) or inner.num_attention_heads
    n_kv_heads = getattr(inner, "n_kv_heads", None) or inner.num_key_value_heads

    # --- Projections + reshape ---
    # Qwen3.5 (Qwen3Next) uses gated attention: q_proj outputs 2x head_dim,
    # split into queries (for attention) + gate (applied after attention).
    q_proj_out = inner.q_proj(x)
    gate = None
    head_dim = inner.k_proj.weight.shape[0] // n_kv_heads
    q_full_head = q_proj_out.shape[-1] // n_heads
    if q_full_head == 2 * head_dim:
        # Gated: split into queries + gate
        q_reshaped = q_proj_out.reshape(B, L, n_heads, q_full_head)
        queries, gate = mx.split(q_reshaped, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
    else:
        queries = q_proj_out.reshape(B, L, n_heads, -1)

    keys = inner.k_proj(x).reshape(B, L, n_kv_heads, -1)
    values = inner.v_proj(x).reshape(B, L, n_kv_heads, -1)

    # Per-head RMSNorm before RoPE (Qwen3, Qwen3.5)
    if hasattr(inner, "q_norm"):
        queries = inner.q_norm(queries)
    if hasattr(inner, "k_norm"):
        keys = inner.k_norm(keys)

    # transpose → (B, heads, L, head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # --- RoPE (per-request position reset) ---
    # mlx_lm uses "rope", mlx_vlm Qwen3.5 uses "rotary_emb"
    if not hasattr(inner, "rope") and not hasattr(inner, "rotary_emb"):
        raise NotImplementedError(
            f"Attention module {type(inner).__name__} does not have a 'rope' "
            "or 'rotary_emb' attribute. Only RoPE-based models are supported."
        )

    queries, keys = apply_packed_rope(
        inner,
        queries,
        keys,
        ctx.cu_seqlens,
        offsets=ctx.offsets if ctx.offsets else None,
    )

    # --- Prepare for gather + native SDPA ---
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Cast keys/values for cache write (3D: L, n_kv_heads, head_dim)
    k_3d = keys[0].transpose(1, 0, 2).astype(kv_cache.dtype)
    v_3d = values[0].transpose(1, 0, 2).astype(kv_cache.dtype)

    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int64)

    # Cast queries to cache dtype for SDPA
    queries = queries.astype(kv_cache.dtype)

    # --- Cache write: MLX-native scatter (pure functional, graph-tracked) ---
    # Flatten cache to [num_slots, num_kv_heads, head_dim], scatter new K/V
    # by slot_mapping, then reshape back.  This creates proper graph nodes
    # that MLX's evaluator can track for dependency ordering and buffer
    # donation — no in-place mutation, no copy_shared_buffer, no const_cast.
    #
    # DONATION INVARIANT: the rebind (below) must drop the list's reference
    # to the old cache *before* mx.eval runs.  At eval time the old cache
    # must have use_count == 1 (only the graph) for MLX to donate its
    # buffer to the scatter output.  Do NOT insert mx.eval between the
    # scatter and the rebind, or hold extra references to the old cache.
    flat_k = kv_cache.key_caches[layer_idx].reshape(-1, kv_cache.num_kv_heads, head_dim)
    flat_k[slot_mapping] = k_3d
    new_k_cache = flat_k.reshape(kv_cache.key_caches[layer_idx].shape)

    flat_v = kv_cache.value_caches[layer_idx].reshape(
        -1, kv_cache.num_kv_heads, head_dim
    )
    flat_v[slot_mapping] = v_3d
    new_v_cache = flat_v.reshape(kv_cache.value_caches[layer_idx].shape)

    # Rebind so next layer / decode step uses the updated cache
    kv_cache.key_caches[layer_idx] = new_k_cache
    kv_cache.value_caches[layer_idx] = new_v_cache

    # --- Attention: per-request gather + mx.fast.scaled_dot_product_attention ---
    # For each request, gather K/V from paged cache into contiguous tensors,
    # then call MLX's native flash attention.  This trades a cheap gather
    # (memory copy) for MLX's fused tiled kernel which is 2-11x faster than
    # the custom Metal paged kernel at all context lengths.
    #
    # queries: (1, n_heads, L, head_dim) — already transposed at line 151
    # new_k_cache/new_v_cache: (num_blocks, block_size, n_kv_heads, head_dim)
    num_requests = len(ctx.cu_seqlens) - 1
    if num_requests == 0:
        out = queries.transpose(0, 2, 1, 3).reshape(B, 0, n_heads * head_dim)
        if gate is not None:
            out = out * mx.sigmoid(gate)
        return inner.o_proj(out)

    # Pre-compute block index arrays once (first SDPA layer), reuse on
    # subsequent layers.  Eliminates (num_layers-1) × num_requests redundant
    # host-to-device mx.array constructions per forward pass.
    if ctx.blocks_mx is None:
        ctx.blocks_mx = []
        for req_idx in range(num_requests):
            ctx_len = ctx.context_lens[req_idx]
            n_blocks = (ctx_len + kv_cache.block_size - 1) // kv_cache.block_size
            block_ids = ctx.block_tables[req_idx]
            ctx.blocks_mx.append(mx.array(block_ids[:n_blocks], dtype=mx.int32))

    outputs = []
    for req_idx in range(num_requests):
        req_start = ctx.cu_seqlens[req_idx]
        req_end = ctx.cu_seqlens[req_idx + 1]
        num_new = req_end - req_start
        ctx_len = ctx.context_lens[req_idx]

        # Query for this request: [1, n_heads, num_new, head_dim]
        rq = queries[:, :, req_start:req_end, :]

        # Gather K/V from paged blocks into contiguous tensors
        blocks = ctx.blocks_mx[req_idx]

        gathered_k = new_k_cache[blocks].reshape(-1, kv_cache.num_kv_heads, head_dim)[
            :ctx_len
        ]
        gathered_v = new_v_cache[blocks].reshape(-1, kv_cache.num_kv_heads, head_dim)[
            :ctx_len
        ]

        # Reshape for SDPA: [1, n_kv_heads, ctx_len, head_dim]
        gathered_k = gathered_k.transpose(1, 0, 2)[None, ...]
        gathered_v = gathered_v.transpose(1, 0, 2)[None, ...]

        # Mask: three modes
        past_len = ctx_len - num_new
        if num_new == 1:
            # Decode: single query attends to all context
            mask = None
        elif past_len == 0:
            # Fresh prefill: use string shorthand (no mask materialization)
            mask = "causal"
        else:
            # Continuation chunk: custom additive mask
            mask = _make_causal_mask(num_new, ctx_len, past_len, rq.dtype)

        out_r = mx.fast.scaled_dot_product_attention(
            rq, gathered_k, gathered_v, scale=inner.scale, mask=mask
        )
        # [1, n_heads, num_new, head_dim] → [1, num_new, n_heads * head_dim]
        out_r = out_r.transpose(0, 2, 1, 3).reshape(1, num_new, -1)
        outputs.append(out_r)

    out = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
    if gate is not None:
        out = out * mx.sigmoid(gate)
    return inner.o_proj(out)
