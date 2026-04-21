# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for compatibility tests."""

import os

import pytest

# Guard optional deps — must come BEFORE bare imports so that pytest
# skips the entire module instead of crashing with ModuleNotFoundError.
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from opaque.core.clipping import clipped_grad
from opaque.functional import make_functional


# NOTE: GPU auto-skip is handled by the top-level conftest.py pytest_runtest_setup hook.


def has_hf_token() -> bool:
    """Return True when a Hugging Face token is available via env vars."""
    return any(
        os.getenv(name)
        for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN")
    )


# Runtime skip for tests that load gated HF models. Not a pytest marker:
# just a skipif mark instance. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN /
# HUGGINGFACE_TOKEN) in the environment to run these tests.
requires_hf_auth = pytest.mark.skipif(
    not has_hf_token(),
    reason="HF token not set (test loads a gated HuggingFace model)",
)


@pytest.fixture(scope="module")
def qwen2_config():
    """Small Qwen2 config for fast testing."""
    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config.num_hidden_layers = 2
    return config


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    """Qwen2 tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


def prepare_lora_model(config, target_modules=None):
    """Helper to create LoRA models.

    Args:
        config: Model config from AutoConfig
        target_modules: List of target modules for LoRA (default: ["q_proj", "v_proj"])

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    model = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    return get_peft_model(model, lora_config)


def run_clipped_grad_test(model, tokenizer, device=None):
    """Helper to run clipped gradient tests.

    Args:
        model: Model to test (should already be on device)
        tokenizer: Tokenizer for the model
        device: Device to use (if None, inferred from model)

    Returns:
        Tuple of (grads, clip_state)
    """
    if device is None:
        device = next(model.parameters()).device

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
        per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
    )
    grads, state = grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )
    return grads, state
