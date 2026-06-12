# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mixtral family (MoE)."""

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
    return build_moe_model("mixtral", device, num_local_experts=8, num_experts_per_tok=2)


def test_mixtral_experts_patched(tiny):
    assert experts_forward_patched(tiny[1])


def test_mixtral_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_mixtral_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_mixtral_vmap_forward(tiny, device):
    assert_vmap_forward(tiny[0], device)


def test_mixtral_vmap_grad(tiny, device):
    assert_vmap_grad(tiny[0], device)
