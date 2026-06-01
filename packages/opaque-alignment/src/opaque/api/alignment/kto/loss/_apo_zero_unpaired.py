# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""APO-zero (unpaired) loss for the KTO family.

Implements the unpaired Anchored Preference Optimisation loss from:

    Karel D'Oosterlinck et al., "Anchored Preference Optimization and
    Contrastive Revisions" (arXiv:2408.06266).

Unlike the paired APO variants in ``loss/dpo/_apo.py`` (which see both a
chosen and a rejected response per example), the unpaired variant takes a
single per-example response together with a boolean ``label`` marking whether
that response is *desirable* (``True``) or *undesirable* (``False``):

    L_i = desirable_weight   · (1 − σ(β · chosen_lr_i))   if label_i else
          undesirable_weight · σ(β · rejected_lr_i)

DP purity: **Tier 1** (§3.3). The output for example *i* depends only on
example *i*'s data — there is no cross-batch aggregate (in particular, no KL
term, in contrast to :func:`kto_loss`). Per-example sensitivity to record swap
is trivially ``O(C)`` after clipping; verified by the NaN-injection contract
test (§11.3).

vmap-safety (§3.4): pure tensor operations only; branching on ``label`` uses
``torch.where`` rather than Python control flow, so the function composes under
``torch.func.vmap(torch.func.grad(...))`` and works on both ``(B,)`` inputs and
0-dim scalars.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["apo_zero_unpaired"]


def apo_zero_unpaired(
    chosen_logratio: torch.Tensor | None,
    rejected_logratio: torch.Tensor | None,
    label: torch.Tensor,
    *,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """Per-example unpaired APO-zero loss (Tier 1, arXiv:2408.06266).

    Selects, per example, between a desirable term and an undesirable term
    according to the boolean ``label``:

        L_i = desirable_weight   · (1 − σ(β · chosen_lr_i))   if label_i else
              undesirable_weight · σ(β · rejected_lr_i)

    There is no KL term, so this is a strict per-example (Tier 1) loss.

    Args:
        chosen_logratio: Per-example log-ratio for a *desirable* response,
            ``log π(y | x) − log π_ref(y | x)``. Shape ``(B,)`` or ``()``. On
            the eager path this may be ``None`` for examples whose label is
            ``False`` (the value is unused there); in that case it is replaced
            with ``torch.zeros_like(rejected_logratio)`` so ``torch.where``
            still type-checks. Under ``vmap`` the caller passes label-masked
            tensors, never ``None``.
        rejected_logratio: Per-example log-ratio for an *undesirable* response.
            Same shape/None semantics as ``chosen_logratio`` (``None`` allowed
            only on the eager path when every label is ``True``).
        label: Per-example boolean tensor; ``True`` marks a desirable response.
            Shape broadcastable to the log-ratios.
        beta: KTO/APO temperature (reference-deviation strength). Positive
            float.
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

    losses_desirable = desirable_weight * (1 - F.sigmoid(beta * chosen_logratio))
    losses_undesirable = undesirable_weight * F.sigmoid(beta * rejected_logratio)

    return torch.where(label.bool(), losses_desirable, losses_undesirable)
