# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different model architectures with vmap/clipped_grad.

Tests Qwen2, Gemma2, DeepSeek, and Phi-2.
"""

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tests.compat.conftest import prepare_lora_model, run_clipped_grad_test


class TestMultiArchitectureCompatibility:
    """Test different model architectures."""

    def test_qwen2_architecture(self, qwen2_config, qwen2_tokenizer):
        """Test Qwen2 architecture (standard MHA/GQA)."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_gemma2_architecture(self):
        """Test Gemma2 architecture (custom sliding window attention)."""
        config = AutoConfig.from_pretrained("google/gemma-2-2b")
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        model = prepare_lora_model(config, target_modules=["q_proj", "v_proj"])
        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0

    @pytest.mark.skip(
        reason="Large model download - enable for full integration testing"
    )
    def test_deepseek_architecture(self):
        """Test DeepSeek architecture."""
        config = AutoConfig.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-base"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_lora_model(config, target_modules=["q_proj", "v_proj"])
        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0

    @pytest.mark.skip(
        reason="Large model download - enable for full integration testing"
    )
    def test_phi2_architecture(self):
        """Test Phi-2 architecture."""
        config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        config.num_hidden_layers = 1
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2", trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_lora_model(config, target_modules=["q_proj", "v_proj"])
        grads, _ = run_clipped_grad_test(model, tokenizer)
        assert len(grads) > 0
