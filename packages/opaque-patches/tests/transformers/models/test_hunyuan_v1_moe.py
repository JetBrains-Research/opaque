# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hunyuan_v1_moe family (MoE)."""

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
    return build_moe_model(
        "hunyuan_v1_moe",
        device,
        num_experts=8,
        moe_topk=2,
        num_experts_per_tok=2,
        head_dim=16,
    )


def test_hunyuan_v1_moe_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_hunyuan_v1_moe_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_hunyuan_v1_moe_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_hunyuan_v1_moe_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_hunyuan_v1_moe_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)
