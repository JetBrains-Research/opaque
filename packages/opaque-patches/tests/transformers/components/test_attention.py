# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different attention implementations with vmap/clipped_grad.

Tests eager and SDPA attention implementations that are compatible with vmap,
including microbatching support and numerical parity.

Known incompatibilities (not tested):
- Flash Attention 2: Uses torch.nonzero which has dynamic output shape
- flex_attention: HigherOrderOperator has no vmap support
"""

import pytest
import torch

from opaque.api.engine.clipping import clipped_grad
from opaque.api.patches.transformers.components.attention import (
    vmap_eager_attention_forward_gemma2,
    vmap_sdpa_attention_forward_gemma2,
)
from opaque.functional import make_functional

from ..._helpers import prepare_lora_model, run_clipped_grad_test


class _Gemma2Attention(torch.nn.Module):
    num_key_value_groups = 1


def _gemma2_softcap_inputs():
    query = torch.tensor([[[[0.25, 0.0], [1.0, 0.0], [4.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]]]])
    value = torch.tensor([[[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]]])
    return query, key, value


def _expected_gemma2_softcap_attention(query, key, value, softcap):
    scores = query @ key.transpose(-2, -1)
    scores = softcap * torch.tanh(scores / softcap)
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


def _assert_gemma2_softcap_attention(attention):
    query, key, value = (tensor.requires_grad_() for tensor in _gemma2_softcap_inputs())
    output, weights = attention(
        _Gemma2Attention(), query, key, value, None, scaling=1.0, softcap=1.0
    )
    ref_query, ref_key, ref_value = (
        tensor.requires_grad_() for tensor in _gemma2_softcap_inputs()
    )
    expected_output, expected_weights = _expected_gemma2_softcap_attention(
        ref_query, ref_key, ref_value, 1.0
    )

    torch.testing.assert_close(output, expected_output.transpose(-3, -2))
    torch.testing.assert_close(weights, expected_weights)
    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_output.square().sum(), (ref_query, ref_key, ref_value)
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


def test_gemma2_softcap_attention_matches_reference():
    """Keep scaled scores near and far beyond the cap before softmax."""
    _assert_gemma2_softcap_attention(vmap_eager_attention_forward_gemma2)
    _assert_gemma2_softcap_attention(vmap_sdpa_attention_forward_gemma2)


class TestAttentionImplementations:
    """Test different attention implementations work with clipped_grad."""

    @pytest.mark.slow
    def test_eager_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention (explicitly patched). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads.pytree) > 0

    def test_sdpa_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention (default, uses patched repeat_kv). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "sdpa"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads.pytree) > 0


class TestAttentionWithMicrobatching:
    """Test attention implementations work with microbatching."""

    def _run_with_microbatch(self, config, tokenizer, device, microbatch_size=2):
        """Helper to run clipped_grad with microbatching."""
        model = prepare_lora_model(config).to(device)

        texts = ["Hello world test", "Another example", "Third sample", "Fourth one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
            microbatch_size=microbatch_size,
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        return grads

    @pytest.mark.slow
    def test_eager_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention with microbatching."""
        qwen2_config._attn_implementation = "eager"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads.pytree) > 0

    @pytest.mark.slow
    def test_sdpa_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention with microbatching."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads.pytree) > 0

    @pytest.mark.slow
    def test_sdpa_microbatch_size_3(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA with microbatch_size=3 (uneven split of batch=4)."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(
            qwen2_config, qwen2_tokenizer, device, microbatch_size=3
        )
        assert len(grads.pytree) > 0


class TestAttentionNumericalParity:
    """Test that SDPA and eager produce similar gradients."""

    @pytest.mark.slow
    def test_sdpa_eager_gradient_parity(self, qwen2_config, qwen2_tokenizer, device):
        """Verify SDPA and eager produce numerically similar clipped gradients.

        Pins SDPA to the ``MATH`` backend so both runs use identical matmul
        order. Without the pin, SDPA picks flash / efficient / math kernels
        based on input shape, device, and dtype — different kernels produce
        different rounding and the comparison becomes flaky at the
        ``rtol=0.2`` tolerance below.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Deterministic init: LoRA and any op that reads torch's default RNG
        # need a fixed seed for a run-to-run-stable comparison.
        torch.manual_seed(0)

        # Run with eager
        qwen2_config._attn_implementation = "eager"
        model_eager = prepare_lora_model(qwen2_config).to(device)
        grads_eager, _ = run_clipped_grad_test(model_eager, qwen2_tokenizer)

        # Run with SDPA (fresh model with same weights — state is copied below)
        torch.manual_seed(0)
        qwen2_config._attn_implementation = "sdpa"
        model_sdpa = prepare_lora_model(qwen2_config).to(device)

        # Copy weights from eager model to SDPA model for fair comparison
        sdpa_state = model_sdpa.state_dict()
        eager_state = model_eager.state_dict()
        for key in sdpa_state:
            if key in eager_state:
                sdpa_state[key] = eager_state[key]
        model_sdpa.load_state_dict(sdpa_state)

        # Pin SDPA to the MATH backend so the backward pass uses the same
        # deterministic reference path eager does (rather than a flash /
        # efficient kernel with different FP rounding).
        with sdpa_kernel(SDPBackend.MATH):
            grads_sdpa, _ = run_clipped_grad_test(model_sdpa, qwen2_tokenizer)

        # Compare gradients - allow for numerical differences between backends.
        # Eager uses manual Q@K matmul; SDPA uses fused CUDA kernels (flash/efficient).
        # These use different algorithms with different floating-point rounding,
        # so we only check that gradients are in the same ballpark.
        assert set(grads_eager.pytree.keys()) == set(grads_sdpa.pytree.keys())
        for key in grads_eager.pytree:
            eager_grad = grads_eager.pytree[key]
            sdpa_grad = grads_sdpa.pytree[key]
            # Wide tolerance: eager uses manual matmul, SDPA uses fused flash/efficient
            # kernels with different FP rounding. Near-zero values can differ by ~2e-3.
            assert torch.allclose(eager_grad, sdpa_grad, rtol=0.2, atol=5e-3), (
                f"Gradient mismatch for {key}: "
                f"max_diff={torch.max(torch.abs(eager_grad - sdpa_grad)).item():.4e}"
            )
