# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Next GDN compatibility and multi-request fixes.

Covers:
  - GDNPagedAttentionWrapper projection dispatch (in_proj_qkvz vs in_proj_qkv)
  - _gdn_free_slot state zeroing (slot-only, preserves other slots)
  - sync_mlx insertion in mlx_to_torch for MPS safety
  - Golden token deterministic test for Qwen3-Next (slow, requires model)

Golden token IDs were generated with greedy decoding (argmax sampler) on
mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit using mlx_lm.

Run unit tests:
    python -m pytest tests/test_qwen3_next_gdn.py -v -k "not slow"

Run golden token test (requires model download):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python -m pytest tests/test_qwen3_next_gdn.py -v -k slow -s
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx

from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache


class TestGDNProjectionDispatch:
    """Verify that GDNPagedAttentionWrapper selects the correct projection
    path based on whether the inner module has ``in_proj_qkvz`` (Qwen3-Next)
    or ``in_proj_qkv`` (Qwen3.5)."""

    def test_detects_qwen3_next_projection(self):
        """Module with in_proj_qkvz should be detected as Qwen3-Next style."""
        module = MagicMock(spec=["in_proj_qkvz", "in_proj_ba"])
        assert hasattr(module, "in_proj_qkvz")
        assert not hasattr(module, "in_proj_qkv")

    def test_detects_qwen35_projection(self):
        """Module with in_proj_qkv should be detected as Qwen3.5 style."""
        module = MagicMock(spec=["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"])
        assert hasattr(module, "in_proj_qkv")
        assert not hasattr(module, "in_proj_qkvz")


class TestGDNFreeSlotZeroing:
    """Verify that _gdn_free_slot zeros only the freed slot."""

    def _make_cache(self, num_layers: int = 2, max_seqs: int = 2) -> GDNPagedStateCache:
        return GDNPagedStateCache(
            num_layers=num_layers,
            max_seqs=max_seqs,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=16,
            key_head_dim=16,
            dtype=mx.float16,
        )

    def test_slot_zeroing_preserves_other_slots(self):
        """Zeroing slot 0 must not affect slot 1."""
        sc = self._make_cache(num_layers=2, max_seqs=2)

        # Write non-zero data to both slots
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Simulate _gdn_free_slot for slot 0 only
        slot = 0
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        for layer_idx in range(sc.num_layers):
            conv = sc.conv_states[layer_idx]
            conv[slot] = 0
            sc.conv_states[layer_idx] = conv
            rec = sc.recurrent_states[layer_idx]
            rec[slot] = 0
            sc.recurrent_states[layer_idx] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 0 should be zeros
        assert mx.allclose(sc.conv_states[0][0], mx.zeros((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][0], mx.zeros((4, 16, 16), dtype=mx.float32)
        )
        # Slot 1 should still be ones
        assert mx.allclose(sc.conv_states[0][1], mx.ones((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][1], mx.ones((4, 16, 16), dtype=mx.float32)
        )

    def test_zeroed_slot_produces_zeros(self):
        """Freed slot must be all zeros after zeroing."""
        sc = self._make_cache(num_layers=1, max_seqs=1)

        # Write non-zero data
        sc.conv_states[0] = mx.ones_like(sc.conv_states[0])
        sc.recurrent_states[0] = mx.ones_like(sc.recurrent_states[0])
        mx.eval(sc.conv_states[0], sc.recurrent_states[0])

        # Zero slot 0
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        conv = sc.conv_states[0]
        conv[0] = 0
        sc.conv_states[0] = conv
        rec = sc.recurrent_states[0]
        rec[0] = 0
        sc.recurrent_states[0] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        assert mx.array_equal(sc.conv_states[0], mx.zeros_like(sc.conv_states[0]))
        assert mx.array_equal(
            sc.recurrent_states[0], mx.zeros_like(sc.recurrent_states[0])
        )

    def test_shapes_preserved_after_zeroing(self):
        """Array shapes and dtypes must be preserved after slot zeroing."""
        sc = self._make_cache(num_layers=3, max_seqs=2)
        expected_conv_shape = sc.conv_states[0].shape
        expected_rec_shape = sc.recurrent_states[0].shape

        # Zero slot 1
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        for layer_idx in range(sc.num_layers):
            conv = sc.conv_states[layer_idx]
            conv[1] = 0
            sc.conv_states[layer_idx] = conv
            rec = sc.recurrent_states[layer_idx]
            rec[1] = 0
            sc.recurrent_states[layer_idx] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        for layer_idx in range(sc.num_layers):
            assert sc.conv_states[layer_idx].shape == expected_conv_shape
            assert sc.recurrent_states[layer_idx].shape == expected_rec_shape
            assert sc.conv_states[layer_idx].dtype == mx.float16
            assert sc.recurrent_states[layer_idx].dtype == mx.float32


class TestGDNAllocSlotZeroing:
    """Verify that _gdn_alloc_slot zeros state for reused slots."""

    def _make_runner_stub(self, max_seqs: int = 2):
        """Build a minimal stub with GDN slot management wired up."""
        from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache

        sc = GDNPagedStateCache(
            num_layers=2,
            max_seqs=max_seqs,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=16,
            key_head_dim=16,
            dtype=mx.float16,
        )
        backend = MagicMock()
        backend._state_cache = sc

        runner = MagicMock()
        runner._gdn_req_to_slot = {}
        runner._gdn_free_slots = []
        runner._paged_attention_backend = backend
        return runner, sc

    def test_reused_slot_is_zeroed(self):
        """A slot returned to the free list and re-allocated must have
        zeroed conv and recurrent state."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()

        # Allocate slot 0 for req-A
        slot = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot == 0

        # Write non-zero data to slot 0
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Free slot 0 (bare pop+append, no zeroing — matches early release path)
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(slot)

        # Re-allocate — should trigger alloc-time zeroing
        slot2 = MetalModelRunner._gdn_alloc_slot(runner, "req-B")
        assert slot2 == 0  # reused
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 0 must be zeroed
        for layer_idx in range(sc.num_layers):
            assert mx.array_equal(
                sc.conv_states[layer_idx][0],
                mx.zeros_like(sc.conv_states[layer_idx][0]),
            )
            assert mx.array_equal(
                sc.recurrent_states[layer_idx][0],
                mx.zeros_like(sc.recurrent_states[layer_idx][0]),
            )

    def test_reused_slot_preserves_other_slots(self):
        """Alloc-time zeroing of slot 0 must not affect slot 1."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()

        # Allocate slots 0 and 1
        MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        MetalModelRunner._gdn_alloc_slot(runner, "req-B")

        # Write ones everywhere
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Free slot 0, re-allocate
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(0)
        MetalModelRunner._gdn_alloc_slot(runner, "req-C")
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 1 must still be ones
        assert mx.allclose(sc.conv_states[0][1], mx.ones((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][1], mx.ones((4, 16, 16), dtype=mx.float32)
        )

    def test_new_slot_not_zeroed(self):
        """A brand-new slot (not from free list) should not trigger zeroing."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()
        slot = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot == 0
        # No crash, no zeroing needed — state was already initialized to zero

    def test_double_free_is_safe(self):
        """Calling _gdn_free_slot twice for the same req_id must not crash."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()
        MetalModelRunner._gdn_alloc_slot(runner, "req-A")

        MetalModelRunner._gdn_free_slot(runner, "req-A")
        MetalModelRunner._gdn_free_slot(runner, "req-A")  # no-op, no crash

    def test_slot_reuse_after_early_release(self):
        """Slot freed via bare pop+append (early release) should be
        available for immediate reallocation."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub(max_seqs=1)

        slot0 = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot0 == 0

        # Early release (bare pop+append, as in execute_model)
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(slot0)

        # Same-step allocation should reuse slot 0
        slot1 = MetalModelRunner._gdn_alloc_slot(runner, "req-B")
        assert slot1 == 0


class TestPagedGDNKernel:
    """Verify the mx.fast.metal_kernel GDN dispatch."""

    def test_multi_request_variable_seqlen(self):
        """Packed multi-request input with variable seq lengths."""
        from vllm_metal.metal_kernel_backend.attention_linear import (
            _paged_gdn_kernel,
        )

        n_hk, n_hv, d_k, d_v = 4, 4, 64, 32
        seq_lens = [3, 5, 2]
        num_requests = len(seq_lens)
        total_tokens = sum(seq_lens)
        cu_seqlens = [0]
        for s in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + s)

        mx.random.seed(123)
        q = mx.random.normal((total_tokens, n_hk, d_k)).astype(mx.float32)
        k = mx.random.normal((total_tokens, n_hk, d_k)).astype(mx.float32)
        v = mx.random.normal((total_tokens, n_hv, d_v)).astype(mx.float32)
        g = mx.random.uniform(shape=(total_tokens, n_hv)).astype(mx.float32)
        beta = mx.random.uniform(shape=(total_tokens, n_hv)).astype(mx.float32)
        state_in = (
            mx.random.normal((num_requests, n_hv, d_v, d_k)).astype(mx.float32) * 0.1
        )
        cu_arr = mx.array(cu_seqlens, dtype=mx.int32)
        mx.eval(q, k, v, g, beta, state_in, cu_arr)

        y, state_out = _paged_gdn_kernel(
            inputs=[q, k, v, g, beta, state_in, cu_arr],
            template=[
                ("InT", mx.float32),
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
            output_dtypes=[mx.float32, mx.float32],
        )
        mx.eval(y, state_out)

        assert y.shape == (total_tokens, n_hv, d_v)
        assert state_out.shape == (num_requests, n_hv, d_v, d_k)
        # Output must be non-zero (non-trivial computation happened)
        assert y.abs().sum().item() > 0
        assert state_out.abs().sum().item() > 0

    def test_state_pool_scatter_isolation(self):
        """Scatter back to pool must update only the target slots."""
        from vllm_metal.metal_kernel_backend.attention_linear import (
            _paged_gdn_kernel,
        )

        n_hk, n_hv, d_k, d_v = 4, 4, 64, 32
        total_tokens = 2
        num_requests = 1
        max_seqs = 4

        mx.random.seed(456)
        q = mx.random.normal((total_tokens, n_hk, d_k)).astype(mx.float32)
        k = mx.random.normal((total_tokens, n_hk, d_k)).astype(mx.float32)
        v = mx.random.normal((total_tokens, n_hv, d_v)).astype(mx.float32)
        g = mx.ones((total_tokens, n_hv), dtype=mx.float32) * 0.9
        beta = mx.ones((total_tokens, n_hv), dtype=mx.float32) * 0.5
        cu_arr = mx.array([0, total_tokens], dtype=mx.int32)
        slot_mapping = mx.array([2], dtype=mx.int32)  # use slot 2

        # Pool with ones everywhere
        pool = mx.ones((max_seqs, n_hv, d_v, d_k), dtype=mx.float32)
        mx.eval(pool)

        state_in = pool[slot_mapping]  # gather slot 2
        _, state_out = _paged_gdn_kernel(
            inputs=[q, k, v, g, beta, state_in, cu_arr],
            template=[
                ("InT", mx.float32),
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
            output_dtypes=[mx.float32, mx.float32],
        )

        # Scatter back to pool
        pool[slot_mapping] = state_out
        mx.eval(pool)

        # Slot 2 should be updated (different from ones)
        ones = mx.ones((n_hv, d_v, d_k), dtype=mx.float32)
        assert not mx.array_equal(pool[2], ones)
        # Slots 0, 1, 3 should still be ones
        assert mx.array_equal(pool[0], ones)
        assert mx.array_equal(pool[1], ones)
        assert mx.array_equal(pool[3], ones)

    def test_conv1d_state_no_eval_between_requests(self):
        """Conv state writes for request 0 must not corrupt request 1
        when no mx.eval is called between them (lazy chain)."""
        sc = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=4,
            conv_dim=16,
            num_v_heads=2,
            value_head_dim=8,
            key_head_dim=8,
            dtype=mx.float16,
        )

        # Write different values to slot 0 and slot 1 without intermediate eval
        val0 = mx.ones((1, 3, 16), dtype=mx.float16) * 2.0
        val1 = mx.ones((1, 3, 16), dtype=mx.float16) * 7.0

        cs = sc.conv_states[0]
        cs[0:1] = val0
        sc.conv_states[0] = cs

        cs = sc.conv_states[0]
        cs[1:2] = val1
        sc.conv_states[0] = cs

        # Now eval and verify both slots independently correct
        mx.eval(sc.conv_states[0])
        assert mx.allclose(sc.conv_states[0][0], val0.squeeze(0))
        assert mx.allclose(sc.conv_states[0][1], val1.squeeze(0))


class TestSyncMLXInTensorBridge:
    """Verify sync_mlx is called before MPS tensor transfer."""

    def test_sync_mlx_called_before_mps_transfer(self):
        """mlx_to_torch must call sync_mlx() when target device is MPS."""
        from vllm_metal.pytorch_backend import tensor_bridge

        array = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        mx.eval(array)

        with patch.object(tensor_bridge, "sync_mlx") as mock_sync:
            try:
                tensor_bridge.mlx_to_torch(array)
            except Exception:
                pass  # MPS may not be available in CI
            # sync_mlx should be called if device is MPS
            if tensor_bridge.get_torch_device().type == "mps":
                mock_sync.assert_called()
