"""§11.2 — vmap-safety sweep over the DPO loss family.

Walks every shipped DPO variant and verifies each survives
``torch.func.vmap(torch.func.grad(...))`` on a 4-example synthetic batch with
finite gradients. Per-variant unit tests check numeric values; this family-wide
meta-test is the single guarantee that no DPO loss silently breaks per-example
gradient composition (plan §3.4 / §11.2).

There is no string registry (mirrors ``opaque.alignment.sft``): the variants are
listed explicitly here, and ``test_sweep_covers_full_family`` guards that the
list stays in sync with the public ``opaque.alignment.dpo.loss`` surface.

SFT exposes direct functions too; its vmap-safety lives in
``tests/sft/loss/test_sft.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

import opaque.alignment.dpo.loss as dpo_loss
from opaque.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    discopop_loss,
    exo_loss,
    hinge_loss,
    ipo_loss,
    nca_loss,
    robust_loss,
    chosen_nll_loss,
    sigmoid_loss,
    sigmoid_norm_loss,
    sppo_loss,
    squarechipo_loss,
)

_BETA = 0.1

# The 14 shipped per-pair DPO variants. ``test_sweep_covers_full_family`` keeps
# this in sync with the public surface; the ``f_divergence_*`` / ``mpo_combine``
# / ``wpo_weights`` / ``ld_dpo_split`` helpers are log-ratio preprocessors, not
# per-pair losses, so they are excluded from the (chosen, rejected) sweep.
_VARIANTS = {
    "sigmoid_loss": sigmoid_loss,
    "hinge_loss": hinge_loss,
    "robust_loss": robust_loss,
    "ipo_loss": ipo_loss,
    "sigmoid_norm_loss": sigmoid_norm_loss,
    "discopop_loss": discopop_loss,
    "chosen_nll_loss": chosen_nll_loss,
    "squarechipo_loss": squarechipo_loss,
    "apo_zero_loss": apo_zero_loss,
    "apo_down_loss": apo_down_loss,
    "exo_loss": exo_loss,
    "nca_loss": nca_loss,
    "bco_loss": bco_loss,
    "sppo_loss": sppo_loss,
}

# Public ``dpo.loss`` names that are NOT per-pair variants on log-ratios: the
# per-sequence logp primitives and the log-ratio combinators.
_NON_VARIANTS = {
    "sequence_logp",
    "fused_sequence_logp",
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
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


def test_sweep_covers_full_family() -> None:
    """The sweep covers every per-pair variant on the public surface.

    Guards against a new variant landing in ``opaque.alignment.dpo.loss``
    without being added to the vmap-safety sweep: the public ``__all__`` minus
    the non-variant names (helpers + fused loss) must equal the swept set.
    """
    public = set(dpo_loss.__all__)
    assert public - _NON_VARIANTS == set(_VARIANTS)
    assert len(_VARIANTS) == 14
