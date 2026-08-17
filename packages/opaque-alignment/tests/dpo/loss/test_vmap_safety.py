"""§11.2 — vmap-safety sweep over the DPO loss family.

Walks every shipped DPO variant and verifies each survives
``torch.func.vmap(torch.func.grad(...))`` on a 4-example synthetic batch with
finite gradients. Per-variant unit tests check numeric values; this family-wide
meta-test is the single guarantee that no DPO loss silently breaks per-example
gradient composition (plan §3.4 / §11.2).

There is no string registry (mirrors ``opaque.alignment.sft``): the variants are
listed explicitly here.

SFT exposes direct functions too; its vmap-safety lives in
``tests/sft/loss/test_nll.py`` and ``tests/sft/loss/test_dft.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    hinge_loss,
    ipo_loss,
    nca_loss,
    robust_loss,
    sigmoid_loss,
    simpo_loss,
    sppo_loss,
)

_BETA = 0.1

# The shipped per-pair DPO variants.
_VARIANTS = {
    "sigmoid_loss": sigmoid_loss,
    "hinge_loss": hinge_loss,
    "robust_loss": robust_loss,
    "ipo_loss": ipo_loss,
    "simpo_loss": simpo_loss,
    "discopop_loss": discopop_loss,
    "chosen_nll_loss": chosen_nll_loss,
    "apo_zero_loss": apo_zero_loss,
    "apo_down_loss": apo_down_loss,
    "exo_loss": exo_loss,
    "nca_loss": nca_loss,
    "bco_loss": bco_loss,
    "sppo_loss": sppo_loss,
}


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_dpo_loss_vmap_grad_finite(name: str) -> None:
    """Every DPO variant: vmap(grad) over per-pair log-ratios is finite."""
    fn = _VARIANTS[name]
    torch.manual_seed(0)
    chosen = torch.randn(4)
    rejected = torch.randn(4)

    def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return fn(c, r, beta=_BETA)

    g_chosen, g_rejected = vmap(grad(per_example, argnums=(0, 1)))(chosen, rejected)
    assert torch.isfinite(g_chosen).all(), f"DPO {name}: non-finite chosen grad"
    assert torch.isfinite(g_rejected).all(), f"DPO {name}: non-finite rejected grad"
