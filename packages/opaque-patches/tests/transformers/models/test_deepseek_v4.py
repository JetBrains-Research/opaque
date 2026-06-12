# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deepseek_v4 family (MoE with custom expert activation).

Experts are left to HF (custom clamped/scaled activation), so only RMSNorm/CE
are patched. HF's experts forward isn't vmap-safe -> forward/backward only.
"""

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
)


@pytest.fixture
def tiny(device):
    return build_moe_model("deepseek_v4", device, )


def test_deepseek_v4_experts_not_patched(tiny):
    # Custom experts are intentionally left to HF (mirrors Liger swiglu=False).
    assert not experts_forward_patched(tiny[1])


def test_deepseek_v4_forward_no_grad(tiny, device):
    assert_forward_no_grad(tiny[0], device)


def test_deepseek_v4_forward_backward(tiny, device):
    assert_forward_backward(tiny[0], device)
