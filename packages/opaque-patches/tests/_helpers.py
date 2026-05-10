# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for patches-only compatibility tests.

DP-SGD ↔ patches integration helpers (``run_clipped_grad_test``) live
in ``integration_tests/dpsgd_patches/_helpers.py``. opaque-patches has
no dependency on opaque-dpsgd; tests that need both belong in the
integration tree.
"""

import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

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
