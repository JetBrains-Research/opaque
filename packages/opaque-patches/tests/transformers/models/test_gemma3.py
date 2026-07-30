# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for gemma3 model."""

import pytest

pytest.importorskip("transformers")

import os
import sys

from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
)

from opaque.patches import apply_model_patches

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_forward,
    assert_vmap_grad,
    get_tiny_config_kwargs,
)


@pytest.fixture
def tiny_model(device):
    kwargs = get_tiny_config_kwargs()
    kwargs.update(
        {
            "head_dim": 16,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "num_hidden_layers": 2,
        }
    )
    config = Gemma3TextConfig(**kwargs)
    config._attn_implementation = "sdpa"
    model = Gemma3ForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_gemma3_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_gemma3_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_gemma3_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_gemma3_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
