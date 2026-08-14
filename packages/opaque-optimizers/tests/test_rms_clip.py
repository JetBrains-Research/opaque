"""Shared update-RMS-clip behaviour across optimizer factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from opaque.optimizers import (
    adadelta,
    adamw,
    ademamix,
    radam,
)


_Factory = Callable[..., tuple[Callable[..., tuple[Any, Any]], Any]]


@pytest.fixture
def params() -> dict[str, torch.Tensor]:
    # Two leaves with very different scales so a global RMS scale is
    # distinguishable from a per-leaf scale.
    return {
        "big": torch.tensor([10.0, -10.0]),
        "small": torch.tensor([0.01, -0.01]),
    }


@pytest.fixture
def grads(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: torch.ones_like(v) for k, v in params.items()}


@pytest.mark.parametrize(
    "factory",
    [
        lambda p, **kw: adamw(p, lr=1e-2, weight_decay=0.0, **kw),
        lambda p, **kw: radam(p, lr=1e-2, weight_decay=0.0, **kw),
        lambda p, **kw: ademamix(p, lr=1e-2, weight_decay=0.0, **kw),
        lambda p, **kw: adadelta(p, lr=1.0, weight_decay=0.0, **kw),
    ],
    ids=["adamw", "radam", "ademamix", "adadelta"],
)
def test_update_rms_clip_uses_one_global_scale(factory: _Factory, params, grads):
    """Clipping multiplies every leaf by the same scalar."""
    step_ref, state_ref = factory(params)
    step_clip, state_clip = factory(params, update_rms_clip=1e-3)

    u_ref, _ = step_ref(grads, state_ref, params=params)
    u_clip, _ = step_clip(grads, state_clip, params=params)

    # Recover the per-leaf scale factors.
    scales = []
    for k in params:
        # Avoid division by ~0.
        mask = u_ref[k].abs() > 1e-12
        if mask.any():
            ratio = (u_clip[k][mask] / u_ref[k][mask]).abs()
            scales.append(float(ratio.mean()))
    assert scales, "expected at least one non-zero reference update"
    # All leaves share one global scale.
    assert max(scales) - min(scales) < 1e-5
