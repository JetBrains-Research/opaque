"""§11.2 — vmap-safety sweep over every public loss registry.

Walks ``DPO_LOSSES`` and ``KTO_LOSSES`` and verifies each entry survives
``torch.func.vmap(torch.func.grad(...))`` on a 4-example synthetic batch with
finite gradients. Per-variant unit tests check numeric values; this
registry-wide meta-test is the single guarantee that no loss in a public
registry silently breaks per-example gradient composition (plan §3.4 / §11.2)
— e.g. a future variant added without its own vmap check is caught here.

(SFT exposes direct functions, not a registry; its vmap-safety is covered in
``tests/sft/loss/test_sft.py``.)
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.func import grad, vmap

from opaque.alignment.loss.dpo import DPO_LOSSES
from opaque.alignment.loss.kto import KTO_LOSSES

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


@pytest.mark.parametrize("name", sorted(KTO_LOSSES))
def test_kto_loss_vmap_grad_finite(name: str) -> None:
    """Every KTO variant: vmap(grad) over per-example log-ratios is finite.

    The Tier-2 ``kto`` variant takes a detached scalar ``kl`` aggregate; it is
    a closure constant here (broadcast, in_dims=None), exactly as the trainer
    computes it once outside the vmap.
    """
    fn = KTO_LOSSES[name]
    params = inspect.signature(fn).parameters
    torch.manual_seed(0)
    chosen = torch.randn(4)
    rejected = torch.randn(4)
    label = torch.tensor([True, False, True, False])
    kl = torch.tensor(0.5)  # detached scalar aggregate (broadcast)

    if "kl" in params:

        def per_example(
            c: torch.Tensor, r: torch.Tensor, lb: torch.Tensor
        ) -> torch.Tensor:
            return fn(c, r, lb, beta=_BETA, kl=kl)
    else:

        def per_example(
            c: torch.Tensor, r: torch.Tensor, lb: torch.Tensor
        ) -> torch.Tensor:
            return fn(c, r, lb, beta=_BETA)

    g_chosen, g_rejected = vmap(grad(per_example, argnums=(0, 1)))(
        chosen, rejected, label
    )
    assert torch.isfinite(g_chosen).all(), f"KTO {name}: non-finite chosen grad"
    assert torch.isfinite(g_rejected).all(), f"KTO {name}: non-finite rejected grad"


def test_registry_coverage_is_complete() -> None:
    """Guard: the sweep covers the full published registries.

    If a new variant is added to a registry, this assertion fails until the
    parametrized sweeps above pick it up — preventing an un-audited loss from
    shipping in a public registry.
    """
    assert len(DPO_LOSSES) == 14
    assert set(KTO_LOSSES) == {"kto", "apo_zero_unpaired"}
