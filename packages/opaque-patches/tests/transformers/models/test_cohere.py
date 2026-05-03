# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for cohere model."""

import pytest

pytest.importorskip("transformers")

from transformers.models.cohere.modeling_cohere import CohereConfig, CohereForCausalLM
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

    config = CohereConfig(**kwargs)
    config._attn_implementation = "eager"
    model = CohereForCausalLM(config).to(device)
    apply_model_patches(model, wrap_eager_attention=True)
    return model


def test_cohere_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_cohere_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_cohere_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_cohere_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
