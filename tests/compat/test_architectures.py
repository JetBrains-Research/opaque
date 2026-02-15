# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different model architectures with vmap/clipped_grad.

Tests Qwen2, Gemma2, DeepSeek, and Phi-2.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional


class TestMultiArchitectureCompatibility:
    """Test different model architectures."""

    def test_qwen2_architecture(self, device):
        """Test Qwen2 architecture (standard MHA/GQA)."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        config._attn_implementation = "eager"
        
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=16, truncation=True)
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
        grads, _ = grad_fn(trainable, frozen, input_ids, attention_mask, labels, state=clip_state)
        assert len(grads) > 0

    @pytest.mark.slow
    def test_gemma2_architecture(self, device):
        """Test Gemma2 architecture (custom sliding window attention).
        
        Note: Requires HuggingFace authentication token for gated model access.
        Run with: huggingface-cli login
        """
        config = AutoConfig.from_pretrained("google/gemma-2-2b")
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config).to(device)
        
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=16, truncation=True)
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
        grads, _ = grad_fn(trainable, frozen, input_ids, attention_mask, labels, state=clip_state)
        assert len(grads) > 0

    @pytest.mark.slow
    def test_deepseek_architecture(self, device):
        """Test DeepSeek architecture (large model download)."""
        config = AutoConfig.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-base"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_lora_model(config, target_modules=["q_proj", "v_proj"]).to(
            device
        )
        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0

    @pytest.mark.slow
    def test_phi2_architecture(self, device):
        """Test Phi-2 architecture (large model download)."""
        config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2", trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_lora_model(config, target_modules=["q_proj", "v_proj"]).to(
            device
        )
        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0

    @pytest.mark.slow
    def test_phi3_architecture(self, device):
        """Test Phi-3 architecture with custom DynamicCache compatibility.
        
        Phi-3 uses a custom DynamicCache implementation and fused QKV projections.
        Tests vmap compatibility patches for Phi-3-specific features.
        
        No authentication required (open model).
        """
        config = AutoConfig.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
        )
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"  # Required for vmap compatibility

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Phi-3 uses fused QKV projection (qkv_proj instead of separate q/k/v)
        model = prepare_lora_model(
            config,
            target_modules=["qkv_proj"],  # Phi-3 fused projection
        ).to(device)

        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0
