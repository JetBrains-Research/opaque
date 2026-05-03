# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for mistral model."""

import pytest

pytest.importorskip("transformers")

from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralForCausalLM,
)
from opaque.patches import apply_model_patches
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (
    get_tiny_config_kwargs,
    assert_forward_no_grad,
    assert_forward_backward,
    assert_vmap_forward,
    assert_vmap_grad,
)


@pytest.fixture
def tiny_model(device):
    kwargs = get_tiny_config_kwargs()

    config = MistralConfig(**kwargs)
    config._attn_implementation = "eager"
    model = MistralForCausalLM(config).to(device)
    apply_model_patches(model, wrap_eager_attention=True)
    return model


def test_mistral_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_mistral_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_mistral_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_mistral_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
