# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test PEFT (Parameter-Efficient Fine-Tuning) methods with vmap/clipped_grad.

Tests LoRA, IA3, Prefix Tuning, P-Tuning, and Prompt Tuning.
"""

from peft import (
    IA3Config,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM
from opaque.patches import apply_model_patches

from ._helpers import prepare_lora_model, run_clipped_grad_test


class TestPEFTMethods:
    """Test different PEFT methods."""

    def test_lora(self, qwen2_config, qwen2_tokenizer, device):
        """Test LoRA (Low-Rank Adaptation)."""
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_ia3(self, qwen2_config, qwen2_tokenizer, device):
        """Test IA3 (Infused Adapter by Inhibiting and Augmenting Inner Activations)."""
        model = AutoModelForCausalLM.from_config(qwen2_config)
        apply_model_patches(model, performance=False, compat=True)

        ia3_config = IA3Config(
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            feedforward_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model.to(device), ia3_config)
        apply_model_patches(model, performance=False, compat=True)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_prefix_tuning(self, qwen2_config, qwen2_tokenizer, device):
        """Test Prefix Tuning."""
        model = AutoModelForCausalLM.from_config(qwen2_config)
        apply_model_patches(model, performance=False, compat=True)

        prefix_config = PrefixTuningConfig(
            num_virtual_tokens=20,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model.to(device), prefix_config)
        apply_model_patches(model, performance=False, compat=True)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_prompt_tuning(self, qwen2_config, qwen2_tokenizer, device):
        """Test Prompt Tuning."""
        model = AutoModelForCausalLM.from_config(qwen2_config)
        apply_model_patches(model, performance=False, compat=True)

        prompt_config = PromptTuningConfig(
            num_virtual_tokens=20,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model.to(device), prompt_config)
        apply_model_patches(model, performance=False, compat=True)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_p_tuning(self, qwen2_config, qwen2_tokenizer, device):
        """Test P-Tuning."""
        model = AutoModelForCausalLM.from_config(qwen2_config)
        apply_model_patches(model, performance=False, compat=True)

        p_tuning_config = PromptEncoderConfig(
            num_virtual_tokens=20,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model.to(device), p_tuning_config)
        apply_model_patches(model, performance=False, compat=True)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0
