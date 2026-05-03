# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for glm4 model."""

import pytest
pytest.importorskip("transformers")

from transformers.models.glm4.modeling_glm4 import Glm4Config, Glm4ForCausalLM
from opaque.patches import apply_model_patches
import sys, os
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

    config = Glm4Config(**kwargs)
    config._attn_implementation = "eager"
    model = Glm4ForCausalLM(config).to(device)
    apply_model_patches(model, wrap_eager_attention=True)
    return model

def test_glm4_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)

def test_glm4_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)

def test_glm4_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)

def test_glm4_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
