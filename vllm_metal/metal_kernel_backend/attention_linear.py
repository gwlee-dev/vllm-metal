# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) with mx.fast.metal_kernel for paged state.

Decomposes the mlx_lm GDN module's forward pass and replaces the recurrent
update step with an mx.fast.metal_kernel that operates on gathered state
slices from a paged pool via slot_mapping.

The kernel participates in MLX's lazy evaluation graph — no explicit mx.eval
barriers are needed in the forward path.  State is gathered from the pool
before the kernel and scattered back afterward, both as lazy MLX ops.

Conv1d remains per-request (stateful), but the expensive recurrent step is
dispatched as a single batched Metal kernel call across all requests.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g

from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import get_context


# ---------------------------------------------------------------------------
# mx.fast.metal_kernel for paged GDN recurrent update
# ---------------------------------------------------------------------------
def _make_paged_gdn_kernel() -> mx.fast.metal_kernel | None:
    """Build an mx.fast.metal_kernel for the paged GDN recurrent update.

    Ported from gdn_linear_attention.metal with adaptations:
    - Functional state (state_in / state_out) instead of in-place pool mutation
    - cu_seqlens for packed variable-length sequences
    - Thread layout matching mlx_lm: grid=(32, Dv, num_requests*Hv),
      threadgroup=(32, 4, 1)
    """
    if not mx.metal.is_available():
        return None

    source = """
        // Thread mapping (matches mlx_lm gated_delta pattern)
        auto n = thread_position_in_grid.z;
        auto req_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // Per-request sequence boundaries from cu_seqlens
        const int seq_start = cu_seqlens[req_idx];
        const int seq_end   = cu_seqlens[req_idx + 1];
        const int seq_len   = seq_end - seq_start;

        // Packed tensor pointers (no batch dim, tokens concatenated)
        // q, k: [total_tokens, Hk, Dk]
        auto q_ = q + seq_start * Hk * Dk + hk_idx * Dk;
        auto k_ = k + seq_start * Hk * Dk + hk_idx * Dk;

        // v, y: [total_tokens, Hv, Dv]
        auto v_ = v + seq_start * Hv * Dv + hv_idx * Dv;
        auto y_ = y + seq_start * Hv * Dv + hv_idx * Dv;

        // state_in, state_out: [num_requests, Hv, Dv, Dk]
        // (gathered from pool before kernel, scattered back after)
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // Load state into registers
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [total_tokens, Hv]  (scalar gating)
        auto g_ = g + seq_start * Hv;
        auto beta_ = beta + seq_start * Hv;

        // Recurrence loop (identical math to mlx_lm gated_delta)
        for (int t = 0; t < seq_len; ++t) {
            float g_val = static_cast<float>(g_[hv_idx]);

            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_val;
                kv_mem += state[i] * static_cast<float>(k_[s_idx]);
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                         * static_cast<float>(beta_[hv_idx]);

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
                out += state[i] * static_cast<float>(q_[s_idx]);
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y_[dv_idx] = static_cast<InT>(out);
            }

            // Advance to next token
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y_ += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }

        // Write state back
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """

    return mx.fast.metal_kernel(
        name="paged_gdn_linear_attention",
        input_names=["q", "k", "v", "g", "beta", "state_in", "cu_seqlens"],
        output_names=["y", "state_out"],
        source=source,
    )


_paged_gdn_kernel = _make_paged_gdn_kernel()


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet).

    Checks for ``conv1d`` (present in all known GatedDeltaNet variants) and
    the absence of ``q_proj`` (which would indicate SDPA).
    """
    return hasattr(module, "conv1d") and not hasattr(module, "q_proj")


class GDNPagedAttentionWrapper(nn.Module):
    """Wraps a GDN linear attention module with C++ Metal kernel dispatch.

    The forward pass decomposes the mlx_lm GDN module into:
    1. Projections (in_proj_qkv, z, a, b) — stateless, batched
    2. Conv1d with state management — per-request (stateful)
    3. Q/K/V split + RMS norm + gating — stateless, batched
    4. Recurrent update — C++ Metal kernel, batched, in-place state pool
    5. Output norm + projection — stateless, batched

    When no ``PagedAttentionContext`` is active, delegates to the original
    module unchanged.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: GDNPagedStateCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gdn_layer_idx", layer_idx)
        object.__setattr__(self, "_gdn_cache_idx", cache_idx)
        object.__setattr__(self, "_gdn_state_cache", state_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
        position_ids: mx.array | None = None,
        **kwargs: Any,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            # GDN is recurrent — does not use position_ids; drop it.
            return self._inner(x, mask=mask, cache=cache)

        inner = self._inner
        cache_idx: int = self._gdn_cache_idx
        state_cache: GDNPagedStateCache = self._gdn_state_cache

        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("GDN wrapper requires cu_seqlens in context")

        num_requests = len(cu_seqlens) - 1
        total_tokens = x.shape[1]

        # === Step 1: Projections (stateless, on full packed input) ===
        if hasattr(inner, "in_proj_qkvz"):
            # Qwen3-Next style: combined projections
            q_pre, k_pre, v_pre, z, b, a = inner.fix_query_key_value_ordering(
                inner.in_proj_qkvz(x), inner.in_proj_ba(x)
            )
            # z: [1, total_tokens, num_v_heads, head_v_dim]
            # b, a: [1, total_tokens, num_v_heads]
            mixed_qkv = mx.concatenate(
                [
                    q_pre.reshape(1, total_tokens, -1),
                    k_pre.reshape(1, total_tokens, -1),
                    v_pre.reshape(1, total_tokens, -1),
                ],
                axis=-1,
            )
        else:
            # Qwen3.5 style: separate projections
            mixed_qkv = inner.in_proj_qkv(x)  # [1, total_tokens, conv_dim]
            z = inner.in_proj_z(x)  # [1, total_tokens, Hv * Dv]
            z = z.reshape(1, total_tokens, -1, inner.head_v_dim)
            b = inner.in_proj_b(x)  # [1, total_tokens, Hv]
            a = inner.in_proj_a(x)  # [1, total_tokens, Hk]

        # === Step 2: Conv1d (batched for decode, per-request for prefill) ===
        # Use stable slot mapping for state pool access.
        slot_ids = (
            ctx.gdn_slot_mapping
            if ctx.gdn_slot_mapping is not None
            else list(range(num_requests))
        )

        # Check if all requests are decode (1 token each) — enables batching
        all_decode = all(
            cu_seqlens[i + 1] - cu_seqlens[i] == 1 for i in range(num_requests)
        )

        if all_decode and num_requests > 1:
            # Batched decode: gather states → concat → single conv1d → scatter
            slot_mapping_conv = mx.array(slot_ids, dtype=mx.int32)
            # gathered: [num_requests, conv_kernel_size-1, conv_dim]
            gathered_states = state_cache.conv_states[cache_idx][slot_mapping_conv]
            # mixed_qkv: [1, num_requests, conv_dim] → [num_requests, 1, conv_dim]
            qkv_batch = mixed_qkv[0, :, :].reshape(num_requests, 1, -1)
            # conv_input: [num_requests, conv_kernel_size, conv_dim]
            conv_input = mx.concatenate([gathered_states, qkv_batch], axis=1)
            # Save new conv state back
            new_states = conv_input[:, -(inner.conv_kernel_size - 1) :]
            pool = state_cache.conv_states[cache_idx]
            pool[slot_mapping_conv] = new_states
            state_cache.conv_states[cache_idx] = pool
            # Single batched conv1d + silu
            conv_out = nn.silu(inner.conv1d(conv_input))
            # Take only output tokens: [num_requests, 1, conv_dim] → [1, num_requests, conv_dim]
            conv_packed = conv_out[:, -1:, :].reshape(1, num_requests, -1)
        else:
            # Per-request loop (prefill with variable lengths, or single request)
            conv_outputs = []
            for req_idx in range(num_requests):
                slot = slot_ids[req_idx]
                start = cu_seqlens[req_idx]
                end = cu_seqlens[req_idx + 1]
                req_qkv = mixed_qkv[:, start:end, :]

                # Load conv state from stable slot
                conv_state = state_cache.conv_states[cache_idx][slot : slot + 1]
                conv_input = mx.concatenate([conv_state, req_qkv], axis=1)

                # Save updated conv state back to stable slot
                new_conv = conv_input[:, -(inner.conv_kernel_size - 1) :]
                cs = state_cache.conv_states[cache_idx]
                cs[slot : slot + 1] = new_conv
                state_cache.conv_states[cache_idx] = cs

                conv_out = nn.silu(inner.conv1d(conv_input))
                # Take only the output tokens (not the conv state prefix)
                conv_outputs.append(conv_out[:, -(end - start) :, :])

            conv_packed = mx.concatenate(conv_outputs, axis=1)

        # === Step 3: Split Q/K/V + norm ===
        q, k, v = [
            t.reshape(1, total_tokens, h, d)
            for t, h, d in zip(
                mx.split(
                    conv_packed,
                    [inner.key_dim, 2 * inner.key_dim],
                    axis=-1,
                ),
                [inner.num_k_heads, inner.num_k_heads, inner.num_v_heads],
                [inner.head_k_dim, inner.head_k_dim, inner.head_v_dim],
                strict=True,
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # === Step 4: Gating (stateless) ===
        # compute_g returns float32; cast to match kernel dispatch dtype.
        g = compute_g(inner.A_log, a, inner.dt_bias).astype(x.dtype)
        beta = mx.sigmoid(b).astype(x.dtype)

        # === Step 5: Lazy Metal kernel — batched recurrent update ===
        n_hk = inner.num_k_heads
        n_hv = inner.num_v_heads
        d_k = inner.head_k_dim
        d_v = inner.head_v_dim

        # Flatten for kernel: remove batch dim.
        # Use float32 for kernel dispatch to avoid float16 overflow in
        # recurrent state accumulation.  Output is cast back after.
        kernel_dtype = mx.float32
        q_flat = q.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype)
        k_flat = k.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype)
        v_flat = v.reshape(total_tokens, n_hv, d_v).astype(kernel_dtype)
        g_flat = g.reshape(total_tokens, n_hv).astype(kernel_dtype)
        beta_flat = beta.reshape(total_tokens, n_hv).astype(kernel_dtype)

        cu_seqlens_arr = mx.array(cu_seqlens, dtype=mx.int32)
        # Stable request → slot mapping from model_runner's allocator.
        if ctx.gdn_slot_mapping is not None:
            slot_mapping = mx.array(ctx.gdn_slot_mapping, dtype=mx.int32)
        else:
            slot_mapping = mx.arange(num_requests, dtype=mx.int32)

        # Gather state from paged pool (lazy)
        recurrent_pool = state_cache.recurrent_states[cache_idx]
        state_in = recurrent_pool[slot_mapping]  # [num_requests, Hv, Dv, Dk]

        # Dispatch lazy kernel — no mx.eval barriers
        y_flat, state_out = _paged_gdn_kernel(
            inputs=[
                q_flat,
                k_flat,
                v_flat,
                g_flat,
                beta_flat,
                state_in,
                cu_seqlens_arr,
            ],
            template=[
                ("InT", kernel_dtype),
                ("StT", mx.float32),
                ("Dk", d_k),
                ("Dv", d_v),
                ("Hk", n_hk),
                ("Hv", n_hv),
            ],
            grid=(32, d_v, num_requests * n_hv),
            threadgroup=(32, 4, 1),
            output_shapes=[
                (total_tokens, n_hv, d_v),
                (num_requests, n_hv, d_v, d_k),
            ],
            output_dtypes=[kernel_dtype, mx.float32],
        )

        # Scatter state back to paged pool (lazy)
        recurrent_pool[slot_mapping] = state_out
        state_cache.recurrent_states[cache_idx] = recurrent_pool

        y_flat = y_flat.astype(x.dtype)

        # === Step 6: Output norm + projection ===
        out = y_flat.reshape(1, total_tokens, n_hv, d_v)
        out = inner.norm(out, z)
        return inner.out_proj(out.reshape(1, total_tokens, -1))
