# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test various training features with vmap/clipped_grad.

Tests gradient checkpointing, mixed precision, and torch.compile.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional


class TestGradientCheckpointing:
    """Test gradient checkpointing compatibility."""

    def test_gradient_checkpointing(self, device):
        """Test gradient checkpointing - should fail due to autograd.Function incompatibility."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        config._attn_implementation = "eager"
        config.use_cache = False  # Required for gradient checkpointing

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config).to(device)
        model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
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

        # Gradient checkpointing uses autograd.Function which is incompatible with vmap
        with pytest.raises(RuntimeError, match="(autograd\\.Function|checkpointing|vmap)"):
            grad_fn, clip_state = clipped_grad(
                per_example_loss, argnums=0, batch_argnums=(2, 3, 4), l2_clip_norm=1.0
            )
            grad_fn(
                trainable, frozen, input_ids, attention_mask, labels, state=clip_state
            )


class TestMixedPrecision:
    """Test mixed precision compatibility."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_precision(self, device, dtype):
        """Test models with mixed precision dtypes."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config).to(device).to(dtype)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
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
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), l2_clip_norm=1.0
        )
        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        assert len(grads) > 0


class TestTorchCompile:
    """Test torch.compile integration."""

    def test_torch_compile(self, device):
        """Test that models work with torch.compile."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config).to(device)

        # Compile the per-example loss function
        def per_example_loss(params, input_ids, mask, labels):
            from opaque import make_functional

            fmodel, _ = make_functional(model, disable_autograd_tracking=True)
            outputs = fmodel(params, input_ids, attention_mask=mask, labels=labels)
            return outputs.loss

        compiled_loss = torch.compile(per_example_loss)

        # This should work (compilation happens lazily)
        assert callable(compiled_loss)
