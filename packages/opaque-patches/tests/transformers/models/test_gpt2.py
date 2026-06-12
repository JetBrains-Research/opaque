# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the gpt2 family (compat-only: batchify + kv_cache, no kernels)."""

import os
import sys

import pytest

pytest.importorskip("transformers")

from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from opaque.patches import apply_model_patches

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (  # noqa: E402
    assert_forward_no_grad,
    assert_forward_backward,
    assert_vmap_forward,
    assert_vmap_grad,
)


@pytest.fixture
def tiny_model(device):
    config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=4,
        # GPT-2 defaults to 0.1 dropout (modern models default 0.0). Active
        # dropout's RNG trips vmap ("same randomness with a batched input") — the
        # only thing that blocked gpt2 under sdpa(vmap). DP training disables
        # dropout anyway, so turn it off and gpt2 runs on sdpa like the rest.
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    config._attn_implementation = "sdpa"
    model = GPT2LMHeadModel(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def test_gpt2_forward_no_grad(tiny_model, device):
    assert_forward_no_grad(tiny_model, device)


def test_gpt2_forward_backward(tiny_model, device):
    assert_forward_backward(tiny_model, device)


def test_gpt2_vmap_forward(tiny_model, device):
    assert_vmap_forward(tiny_model, device)


def test_gpt2_vmap_grad(tiny_model, device):
    assert_vmap_grad(tiny_model, device)
