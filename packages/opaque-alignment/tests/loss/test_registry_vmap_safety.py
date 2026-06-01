"""§11.2 — vmap-safety sweep over the DPO loss registry.

Walks ``DPO_LOSSES`` and verifies each entry survives
``torch.func.vmap(torch.func.grad(...))`` on a 4-example synthetic batch with
finite gradients. Per-variant unit tests check numeric values; this
registry-wide meta-test is the single guarantee that no loss in the registry
silently breaks per-example gradient composition (plan §3.4 / §11.2).

SFT exposes direct functions (no registry); its vmap-safety lives in
``tests/sft/loss/test_sft.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.alignment.loss.dpo import DPO_LOSSES

_BETA = 0.1


@pytest.mark.parametrize("name", sorted(DPO_LOSSES))
def test_dpo_loss_vmap_grad_finite(name: str) -> None:
    """Every DPO variant: vmap(grad) over per-pair log-ratios is finite."""
    fn = DPO_LOSSES[name]
    torch.manual_seed(0)
    chosen = torch.randn(4)
    rejected = torch.randn(4)

    def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return fn(c, r, beta=_BETA)

    g_chosen, g_rejected = vmap(grad(per_example, argnums=(0, 1)))(chosen, rejected)
    assert torch.isfinite(g_chosen).all(), f"DPO {name}: non-finite chosen grad"
    assert torch.isfinite(g_rejected).all(), f"DPO {name}: non-finite rejected grad"


def test_registry_coverage_is_complete() -> None:
    """Guard: the sweep covers the full published DPO registry."""
    assert len(DPO_LOSSES) == 14
