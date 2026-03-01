"""Lightweight CUDA smoke tests for CI stability.

These tests intentionally use tiny tensors and avoid gated external models.
"""

import importlib.util

import pytest
import torch

from opaque.compat.kernels import opaque_cross_entropy_loss, opaque_swiglu


@pytest.mark.cuda
def test_cuda_swiglu_smoke():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton not installed")

    gate = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    up = torch.randn(4, 8, device="cuda", dtype=torch.float16)

    out = opaque_swiglu(gate, up)

    assert out.device.type == "cuda"
    assert out.shape == gate.shape
    assert torch.isfinite(out).all()


@pytest.mark.cuda
def test_cuda_cross_entropy_smoke():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton not installed")

    logits = torch.randn(2, 3, 16, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, 16, (2, 3), device="cuda")

    loss = opaque_cross_entropy_loss(logits, labels)

    assert loss.device.type == "cuda"
    assert loss.shape == labels.shape
    assert torch.isfinite(loss).all()
