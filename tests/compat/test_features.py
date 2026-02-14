# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test various training features with vmap/clipped_grad.

Tests gradient checkpointing, mixed precision, and torch.compile.
"""

import pytest
import torch

from tests.compat.conftest import prepare_lora_model, run_clipped_grad_test


class TestGradientCheckpointing:
    """Test gradient checkpointing compatibility."""

    def test_gradient_checkpointing(self, qwen2_config, qwen2_tokenizer, device):
        """Test gradient checkpointing - should fail due to autograd.Function incompatibility."""
        qwen2_config._attn_implementation = "eager"
        qwen2_config.use_cache = False  # Required for gradient checkpointing
        model = prepare_lora_model(qwen2_config).to(device)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Gradient checkpointing uses autograd.Function which is incompatible with vmap
        # unless it overrides setup_context staticmethod
        with pytest.raises(RuntimeError, match="autograd.Function"):
            grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)


class TestMixedPrecision:
    """Test mixed precision compatibility."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_precision(self, qwen2_config, qwen2_tokenizer, device, dtype):
        """Test models with mixed precision dtypes."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config).to(device).to(dtype)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0


class TestTorchCompile:
    """Test torch.compile integration."""

    def test_torch_compile(self, qwen2_config, qwen2_tokenizer, device):
        """Test that models work with torch.compile."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config).to(device)

        # Compile the per-example loss function
        def per_example_loss(trainable_params, frozen_params, input_ids, mask, labels):
            from opaque import make_functional

            fmodel, _, _ = make_functional(model, disable_autograd_tracking=True)
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, input_ids, attention_mask=mask, labels=labels)
            return outputs.loss

        compiled_loss = torch.compile(per_example_loss)

        # Execute the compiled function once to ensure compilation works
        from opaque import make_functional

        fmodel, trainable_params, frozen_params = make_functional(
            model, disable_autograd_tracking=True
        )

        encoded = qwen2_tokenizer("test input", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        mask = encoded.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(input_ids)
        else:
            mask = mask.to(device)
        labels = input_ids.clone()

        loss = compiled_loss(trainable_params, frozen_params, input_ids, mask, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
