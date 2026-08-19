# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for opaque-patches compatibility tests.

The helpers exercise ``opaque.patches.apply_model_patches`` against
``opaque.api.engine.clipping.clipped_grad`` to verify that patched HF
models still vmap-grad cleanly. These tests intentionally stick to the
**engine** clipping primitive (rather than ``opaque.dpsgd.clipping``)
because patches has no dependency on opaque-dpsgd; the test subject is
"patches preserve gradient flow under vmap", not any DP-SGD specific
behavior.

Genuine DP-SGD ↔ patches and DP-FTRL ↔ patches integration tests
(end-to-end pipelines exercising noise + accounting in addition to
clipping) live in ``tests/integration/patches/`` if and when they are
added.
"""

import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque.api.engine.clipping import clipped_grad
from opaque.patches import apply_model_patches
from opaque.torch.functional import make_functional


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
    # Shrink to a tiny-but-structurally-faithful Qwen2: 2 layers and a narrow
    # width keep the vmap(grad) attention/MLP work cheap, while a 4-head /
    # 2-kv-head split preserves the GQA ``repeat_kv`` path these tests
    # exercise. ``head_dim`` is pinned explicitly so it can't stay at the
    # 0.5B value (64) and desync q_proj's output width from ``hidden_size``.
    # ``vocab_size`` is left untouched so the real Qwen2 tokenizer's ids stay
    # in range (the embedding is frozen, so its size costs only a lookup).
    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = config.hidden_size // config.num_attention_heads
    config.intermediate_size = 256
    return config


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


def prepare_lora_model(config, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    model = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    apply_model_patches(
        model,
        performance=False,
        compat=True,
        lora=True,
        activation=False,
        rms_norm=False,
        rope=False,
        cross_entropy=False,
        eager_attention=True,
    )
    return model


def run_clipped_grad_test(model, tokenizer, device=None):
    """Verify a patched model still vmap-grad-clips correctly.

    Engine-side assertion — uses ``opaque.api.engine.clipping.clipped_grad``
    rather than ``opaque.dpsgd.clipping`` so it doesn't pull in any
    DP-SGD-specific behavior. The test passes if the model produces
    finite gradients of the right shape under ``vmap(grad(...))``.
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
