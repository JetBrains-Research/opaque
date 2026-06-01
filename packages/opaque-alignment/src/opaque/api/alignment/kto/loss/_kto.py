# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""KTO (Kahneman-Tversky Optimisation) loss — Tier-2, detached batch-mean KL.

Implements the single loss prescribed by the KTO paper:

    Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe
    Kiela, "KTO: Model Alignment as Prospect Theoretic Optimization"
    (arXiv:2402.01306). Eq. (8) defines the per-example loss as

        L_KTO(i) = w(y_i) · (1 − v(x_i, y_i))

      v(x, y) = σ( β · (r_θ(x, y) − z_0))   for desirable y
                σ( β · (z_0 − r_θ(x, y)))   for undesirable y

    where r_θ(x, y) = log π_θ(y|x) − log π_ref(y|x) is the implicit reward
    (here the per-example ``chosen_logratio`` / ``rejected_logratio``) and
    ``z_0`` is the batch KL term ``KL(π_θ ‖ π_ref)``. Critically, the paper
    states "we do not back-propagate through z_0" — z_0 is a **stop-gradient**
    (detached) batch aggregate. This matches TRL ``kto_trainer.py:882-884``,
    where ``kl`` is gathered, mean-reduced, and ``.detach()``-ed before the
    per-example loss is formed.

DP purity: **Tier 2** (§3.3, §8.1). The output for example *i* depends on
example *i*'s own log-ratio and on the scalar ``kl`` aggregate that the caller
computes once over the microbatch — **outside** the per-example ``vmap`` — and
``.detach()``-es before broadcasting in. Because ``kl`` is a detached batch
*mean*, swapping a single record changes it by ``O(1/n)``, and the per-example
loss's sensitivity to that swap is therefore ``O(1/n)`` rather than ``O(1)``
(the §3.3 Tier-2 condition; theoretical basis: Kumar et al., NeurIPS 2023,
arXiv:2310.03104). The detach keeps ``kl`` out of the released gradient's
autograd graph, so it does not add to the privacy ledger. Verified by the
aggregate-detach audit + bounded-leverage test (§11.4).

Leverage bound: ``|∂L_i/∂kl| = w · β · σ'(·) ≤ β · max(weights) · 1/4`` (since
``σ'(·) = σ(1−σ) ≤ 1/4``), hence ``≤ β · max(weights)``.

vmap-safety (§3.4): pure tensor operations only; the desirable/undesirable
branch is selected with ``torch.where`` (no Python control flow on tensor
values); the function works on ``(B,)`` inputs and 0-dim scalars and composes
under ``torch.func.vmap(torch.func.grad(...))``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["kto_loss"]


def kto_loss(
    chosen_logratio: torch.Tensor | None,
    rejected_logratio: torch.Tensor | None,
    label: torch.Tensor,
    *,
    beta: float,
    kl: torch.Tensor,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """Per-example KTO loss (Tier 2, arXiv:2402.01306 Eq. 8).

    Selects, per example, between a desirable and an undesirable term using the
    boolean ``label``, with the batch KL term ``kl`` (= ``z_0``) entering both
    branches as a detached, stop-gradient scalar:

        L_i = desirable_weight   · (1 − σ(β · (chosen_lr_i − kl)))   if label_i
              undesirable_weight · (1 − σ(β · (kl − rejected_lr_i))) otherwise

    Args:
        chosen_logratio: Per-example log-ratio for a *desirable* response,
            ``log π(y | x) − log π_ref(y | x)``. Shape ``(B,)`` or ``()``. On
            the eager path this may be ``None`` for examples whose label is
            ``False`` (the value is unused there); it is then replaced with
            ``torch.zeros_like(rejected_logratio)`` so ``torch.where`` still
            type-checks. Under ``vmap`` the caller passes label-masked tensors,
            never ``None``.
        rejected_logratio: Per-example log-ratio for an *undesirable* response.
            Same shape/None semantics as ``chosen_logratio``.
        label: Per-example boolean tensor; ``True`` marks a desirable response.
            Shape broadcastable to the log-ratios.
        beta: KTO temperature (reference-deviation strength). Positive float.
        kl: **Scalar, detached** batch-mean KL term ``z_0`` (broadcast to every
            example). The caller computes it once over the active microbatch,
            **outside** the per-example ``vmap`` — typically as
            ``(policy_KL_logps - ref_KL_logps).mean().detach().clamp(min=0)`` —
            and broadcasts it in (the Tier-2 contract, §8.1). It MUST be
            detached before being passed; the aggregate-detach audit (§11.4)
            enforces this. Under a Poisson batch of size ≤ 1 the caller passes
            ``kl=0`` (degenerating to an ``apo_zero_unpaired``-like step for
            that microbatch — see §9.4).
        desirable_weight: Weight applied to the desirable term. Defaults to
            ``1.0``.
        undesirable_weight: Weight applied to the undesirable term. Defaults to
            ``1.0``.

    Returns:
        Per-example loss tensor of the same shape as the inputs.
    """
    # Eager path: one side may be None when that side is unused. Substitute a
    # zero tensor with the same shape/dtype as the other side so torch.where
    # type-checks. Under vmap the caller passes label-masked tensors (not None).
    if chosen_logratio is None:
        chosen_logratio = torch.zeros_like(rejected_logratio)
    if rejected_logratio is None:
        rejected_logratio = torch.zeros_like(chosen_logratio)

    # KTO paper Eq. (8): z_0 (= kl) is a stop-gradient batch aggregate. The
    # caller is responsible for the .detach(); we do not back-propagate through
    # it here either way.
    losses_desirable = desirable_weight * (1 - F.sigmoid(beta * (chosen_logratio - kl)))
    losses_undesirable = undesirable_weight * (
        1 - F.sigmoid(beta * (kl - rejected_logratio))
    )

    return torch.where(label.bool(), losses_desirable, losses_undesirable)
