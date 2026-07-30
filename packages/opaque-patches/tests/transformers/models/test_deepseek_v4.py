# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deepseek_v4 family (MoE with custom expert activation).

Experts are intentionally left to HF (scaled experts + interleaved partial RoPE),
so Opaque patches only RMSNorm/CE. HF's own experts forward is built on
``torch._grouped_mm``, which is vmap-traceable — so DP-SGD ``vmap(grad)`` works
through the unpatched experts, and there's no need for an Opaque expert kernel.
"""

import os
import sys

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (
    assert_forward_backward,
    assert_forward_no_grad,
    assert_vmap_grad,
    build_moe_model,
    experts_forward_patched,
)


@pytest.fixture
def tiny(device):
    return build_moe_model("deepseek_v4", device)


def test_deepseek_v4_experts_not_patched(tiny):
    # Custom experts are intentionally left to HF (mirrors Liger swiglu=False);
    # HF's grouped-GEMM forward is already fast and vmap-safe.
    assert not experts_forward_patched(tiny[1])


def test_deepseek_v4_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_deepseek_v4_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)


def test_deepseek_v4_vmap_grad(tiny, device):
    """DP-SGD per-sample gradients run through HF's (vmap-safe) experts forward."""
    assert_vmap_grad(tiny[0], device)
