"""Shared helpers for DP-SGD ↔ patches integration tests.

These helpers exercise ``opaque.patches.apply_model_patches`` against
``opaque.dpsgd.clipping.clipped_grad`` to verify that patched HF models
still vmap-grad cleanly. They live here (not in
``packages/opaque-patches/tests/``) because patches doesn't depend on
dpsgd; this is the natural integration parking spot.

Patches-only fixtures (``qwen2_config``, ``qwen2_tokenizer``,
``requires_hf_auth``, ``prepare_lora_model``) live in
``packages/opaque-patches/tests/_helpers.py`` and are re-exported here
for the tests that need both the patches preflight and the dpsgd
gradient check.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from opaque.dpsgd.clipping import clipped_grad
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
