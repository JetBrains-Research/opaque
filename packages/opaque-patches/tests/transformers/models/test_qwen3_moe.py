# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_moe family (MoE)."""

import os
import sys

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (  # noqa: E402
    build_moe_model,
    experts_forward_patched,
    assert_forward_no_grad,
    assert_forward_backward,
    assert_vmap_forward,
    assert_vmap_grad,
)


@pytest.fixture
def tiny(device):
    return build_moe_model("qwen3_moe", device, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, decoder_sparse_step=1, norm_topk_prob=True)


def test_qwen3_moe_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_qwen3_moe_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_qwen3_moe_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_qwen3_moe_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_qwen3_moe_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)
