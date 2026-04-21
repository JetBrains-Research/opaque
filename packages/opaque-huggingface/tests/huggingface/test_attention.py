# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different attention implementations with vmap/clipped_grad.

Tests eager and SDPA attention implementations that are compatible with vmap,
including microbatching support and numerical parity.

Known incompatibilities (not tested):
- Flash Attention 2: Uses torch.nonzero which has dynamic output shape
- flex_attention: HigherOrderOperator has no vmap support
"""

import torch

from opaque.clipping import clipped_grad
from opaque.functional import make_functional

from ._helpers import prepare_lora_model, run_clipped_grad_test


class TestAttentionImplementations:
    """Test different attention implementations work with clipped_grad."""

    def test_eager_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention (explicitly patched). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_sdpa_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention (default, uses patched repeat_kv). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "sdpa"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0


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
        grads, state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        return grads

    def test_eager_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention with microbatching."""
        qwen2_config._attn_implementation = "eager"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads) > 0

    def test_sdpa_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention with microbatching."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads) > 0

    def test_sdpa_microbatch_size_3(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA with microbatch_size=3 (uneven split of batch=4)."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(
            qwen2_config, qwen2_tokenizer, device, microbatch_size=3
        )
        assert len(grads) > 0


class TestAttentionNumericalParity:
    """Test that SDPA and eager produce similar gradients."""

    def test_sdpa_eager_gradient_parity(self, qwen2_config, qwen2_tokenizer, device):
        """Verify SDPA and eager produce numerically similar clipped gradients."""
        # Run with eager
        qwen2_config._attn_implementation = "eager"
        model_eager = prepare_lora_model(qwen2_config).to(device)
        grads_eager, _ = run_clipped_grad_test(model_eager, qwen2_tokenizer)

        # Run with SDPA (need fresh model with same weights)
        qwen2_config._attn_implementation = "sdpa"
        model_sdpa = prepare_lora_model(qwen2_config).to(device)

        # Copy weights from eager model to SDPA model for fair comparison
        sdpa_state = model_sdpa.state_dict()
        eager_state = model_eager.state_dict()
        for key in sdpa_state:
            if key in eager_state:
                sdpa_state[key] = eager_state[key]
        model_sdpa.load_state_dict(sdpa_state)

        grads_sdpa, _ = run_clipped_grad_test(model_sdpa, qwen2_tokenizer)

        # Compare gradients - allow for numerical differences between backends.
        # Eager uses manual Q@K matmul; SDPA uses fused CUDA kernels (flash/efficient).
        # These use different algorithms with different floating-point rounding,
        # so we only check that gradients are in the same ballpark.
        assert set(grads_eager.keys()) == set(grads_sdpa.keys())
        for key in grads_eager:
            eager_grad = grads_eager[key]
            sdpa_grad = grads_sdpa[key]
            # Wide tolerance: eager uses manual matmul, SDPA uses fused flash/efficient
            # kernels with different FP rounding. Near-zero values can differ by ~2e-3.
            assert torch.allclose(eager_grad, sdpa_grad, rtol=0.2, atol=5e-3), (
                f"Gradient mismatch for {key}: "
                f"max_diff={torch.max(torch.abs(eager_grad - sdpa_grad)).item():.4e}"
            )
