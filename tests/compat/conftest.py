# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for compatibility tests.

Dependencies: Install with `uv sync --group compat`
"""

import pytest
import torch

pytest.importorskip("peft", reason="peft not installed, run `uv sync --group compat`")
pytest.importorskip("transformers", reason="transformers not installed")

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional

# Mark entire compat directory as requiring compat dependencies
pytestmark = pytest.mark.compat


# device fixture is inherited from tests/conftest.py
# It automatically selects: CUDA > MPS > CPU


@pytest.fixture
def qwen2_config():
    """Small Qwen2 config for testing."""
    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2  # GQA
    return config


@pytest.fixture
def qwen2_tokenizer():
    """Qwen2 tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


def prepare_lora_model(config, target_modules=None):
    """Helper to create a model with LoRA applied."""
    model = AutoModelForCausalLM.from_config(config)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    return get_peft_model(model, lora_config)


def run_clipped_grad_test(model, tokenizer):
    """Helper to run a basic clipped_grad test.

    Returns:
        tuple: (gradients_dict, clip_state)
    """
    # Determine device from model
    device = next(model.parameters()).device

    # Prepare batch
    texts = ["Hello world test", "Another example", "Third sample", "Final one"]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, max_length=16, truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()

    # Convert to functional
    fmodel, trainable, frozen = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )

    def per_example_loss(
        trainable_params, frozen_params, input_ids_single, mask_single, labels_single
    ):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(
            all_params,
            input_ids_single,
            attention_mask=mask_single,
            labels=labels_single,
        )
        return outputs.loss

    # Create clipped gradient function
    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        l2_clip_norm=1.0,
    )

    # Compute gradients
    grads, new_state = grad_fn(
        trainable,
        frozen,
        input_ids,
        attention_mask,
        labels,
        state=clip_state,
    )

    return grads, new_state
