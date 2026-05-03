# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for cohere2 model."""

import pytest
pytest.importorskip("transformers")

from transformers.models.cohere2.modeling_cohere2 import Cohere2Config, Cohere2ForCausalLM
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

    config = Cohere2Config(**kwargs)
    config._attn_implementation = "eager"
    model = Cohere2ForCausalLM(config).to(device)
    apply_model_patches(model, wrap_eager_attention=True)
    return model

def test_cohere2_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)

def test_cohere2_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)

def test_cohere2_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)

def test_cohere2_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
