"""Torch-native ``clipped_fun`` compute-dtype behavior."""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opaque.api.engine.clipping._clipped_fun import clipped_fun
from opaque.types import ClippedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


# ----------------------------------------------------------------------------
# clipped_fun: dtype (output) vs compute_dtype (internal) split
# ----------------------------------------------------------------------------


def _loss_fn(params, x):
    """Simple per-example loss for vmap'd grad."""
    return ((params * x) ** 2).sum()


def test_clipped_fun_output_dtype_default_matches_input():
    """dtype=None → output dtype matches function output dtype."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(per_example, batch_argnums=1, clipping_norm=1.0)
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.bfloat16


def test_clipped_fun_output_dtype_overridable():
    """dtype=fp32 → caller gets fp32 sum back regardless of input."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        dtype=torch.float32,
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.float32


def test_clipped_fun_compute_dtype_independent_of_output():
    """compute_dtype is internal-only; default dtype keeps output type-stable."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        compute_dtype=torch.float32,  # accumulate in fp32
        # dtype left as None → output matches input (bf16)
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.bfloat16


def test_clipped_fun_microbatch_honors_compute_dtype():
    """microbatch path also honors compute_dtype."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        microbatch_size=2,
        compute_dtype=torch.float32,
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(6, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    # default dtype=None → output stays bf16; the fp32 compute happens internally
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("compute_dtype", [None, torch.float32, torch.float64])
def test_clipped_fun_smoke_compute_dtype_variants(compute_dtype):
    """Each compute_dtype option produces a finite, sensible result."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        compute_dtype=compute_dtype,
    )
    params = torch.tensor([1.0, 2.0])
    x = torch.randn(4, 2)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert torch.isfinite(result).all()
