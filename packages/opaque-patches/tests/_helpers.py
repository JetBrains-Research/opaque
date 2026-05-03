# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for compatibility tests."""

import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from opaque.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches import apply_model_patches


def has_hf_token() -> bool:
    return any(
        os.getenv(name)
        for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN")
    )


requires_hf_auth = pytest.mark.skipif(
    not has_hf_token(),
    reason="HF token not set (test loads a gated HuggingFace model)",
)


@pytest.fixture(scope="module")
def qwen2_config():
    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config.num_hidden_layers = 2
    return config


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


def prepare_lora_model(config, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    model = AutoModelForCausalLM.from_config(config)
    apply_model_patches(
        model,
        fuse_swiglu=False,
        fuse_rms_norm=False,
        fuse_rope=False,
        fuse_cross_entropy=False,
        wrap_eager_attention=True,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    return get_peft_model(model, lora_config)


def run_clipped_grad_test(model, tokenizer, device=None):
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
