# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""f-divergence log-ratio remaps for the DPO family (plan §7.1).

The standard DPO objective (Rafailov et al., 2023) implicitly minimises the
*reverse* KL divergence between the policy and the reference. The DPO loss can
be generalised to other f-divergences by remapping each side's per-example
log-ratio ``g(logratio)`` *before* the preference logits are formed, i.e.::

    delta = g(chosen_logratio) - g(rejected_logratio)

This remapped ``delta`` then feeds any DPO variant (sigmoid, hinge, ipo, ...)
unchanged. The four supported f-divergences and their per-side remap ``g`` are
(plan §7.1, "DPO-specific helpers"):

    reverse_kl        g(x) = x                          (identity — the DPO default)
    forward_kl        g(x) = -exp(-x)
    js_divergence     g(x) = logsigmoid(x)
    alpha_divergence  g(x) = exp((alpha - 1) * x) / (alpha - 1)   (alpha != 1)

The ``alpha_divergence`` remap is undefined at ``alpha == 1`` (a removable
singularity whose limit *is* the ``reverse_kl`` identity); callers must use
``reverse_kl`` there, and this module raises ``ValueError`` to make that
explicit rather than dividing by zero.

The ``exp`` paths (``forward_kl`` and ``alpha_divergence``) clamp the exponent
through :func:`_cap_exp` so that large log-ratios do not overflow to ``inf``
in low-precision dtypes (fp16/bf16).

DP purity: **Tier 1** (§3.3). The remap of example *i* depends only on example
*i*'s log-ratio; there is no cross-example aggregate.

vmap-safety (§3.4): pure tensor operations only. Dispatch on ``f_divergence_type``
and ``alpha`` uses ordinary Python ``if``/``elif`` — these are *static* Python
arguments, not traced tensor values, so branching on them is vmap-safe (no
control flow on tensor values, no ``torch.where`` needed). No module state, no
``.item()``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

__all__ = ["f_divergence_remap", "f_divergence_logits"]

FDivergence = Literal["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]


def _cap_exp(x: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    """``exp`` with an upper clamp on the exponent to avoid fp16/bf16 overflow.

    ``exp`` grows extremely fast: in bf16 it overflows to ``inf`` for exponents
    above roughly ``88``, and the limited precision means even smaller exponents
    can saturate when accumulated. Clamping the *exponent* (not the result) at
    ``cap`` keeps the output finite while leaving the small/moderate range — the
    region the gradient actually cares about — untouched.

    Args:
        x: Exponent tensor.
        cap: Upper bound applied to ``x`` before exponentiation. Defaults to
            ``20.0`` (``exp(20) ~ 4.85e8``, comfortably finite in fp16/bf16).

    Returns:
        ``exp(clamp(x, max=cap))``, same shape and dtype as ``x``.
    """
    return torch.exp(torch.clamp(x, max=cap))


def f_divergence_remap(
    logratio: torch.Tensor,
    *,
    f_divergence_type: FDivergence = "reverse_kl",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Remap a per-example log-ratio under the chosen f-divergence (§7.1).

    Applies the per-side remap ``g`` for the requested f-divergence:

    - ``reverse_kl``: ``g(x) = x`` (identity — grad-transparent).
    - ``forward_kl``: ``g(x) = -_cap_exp(-x)``.
    - ``js_divergence``: ``g(x) = logsigmoid(x)``.
    - ``alpha_divergence``: ``g(x) = _cap_exp((alpha - 1) * x) / (alpha - 1)``;
      requires ``alpha != 1`` (the ``alpha == 1`` limit is ``reverse_kl``).

    The function is elementwise and works on batched ``(B,)`` inputs as well as
    0-dim scalars, making it safe to call under
    ``torch.func.vmap(torch.func.grad(...))``.

    Args:
        logratio: Per-example log-ratio ``log π(y | x) − log π_ref(y | x)``.
            Shape ``(B,)`` or ``()``.
        f_divergence_type: Which f-divergence remap to apply. One of
            ``"reverse_kl"``, ``"forward_kl"``, ``"js_divergence"``,
            ``"alpha_divergence"``. Defaults to ``"reverse_kl"``.
        alpha: The α parameter for ``alpha_divergence``. Must not equal ``1``
            (that case is ``reverse_kl``). Ignored by the other divergences.
            Defaults to ``1.0``.

    Returns:
        Remapped log-ratio tensor of the same shape as ``logratio``.

    Raises:
        ValueError: If ``f_divergence_type == "alpha_divergence"`` and
            ``alpha == 1`` (use ``"reverse_kl"`` instead), or if
            ``f_divergence_type`` is not a recognised divergence name.
    """
    if f_divergence_type == "reverse_kl":
        # Identity: returned unchanged so the gradient path is fully
        # transparent (d g/d x = 1), reproducing the standard DPO objective.
        return logratio
    if f_divergence_type == "forward_kl":
        return -_cap_exp(-logratio)
    if f_divergence_type == "js_divergence":
        return F.logsigmoid(logratio)
    if f_divergence_type == "alpha_divergence":
        if alpha == 1:
            raise ValueError(
                "alpha_divergence is undefined at alpha == 1 (its limit is the "
                "reverse_kl identity); call f_divergence_remap(..., "
                'f_divergence_type="reverse_kl") instead.'
            )
        return _cap_exp((alpha - 1) * logratio) / (alpha - 1)
    raise ValueError(
        f"Unknown f_divergence_type {f_divergence_type!r}; expected one of "
        '"reverse_kl", "forward_kl", "js_divergence", "alpha_divergence".'
    )


def f_divergence_logits(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    f_divergence_type: FDivergence = "reverse_kl",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Form the remapped preference logits ``delta`` under an f-divergence (§7.1).

    Computes::

        delta = f_divergence_remap(chosen_logratio) - f_divergence_remap(rejected_logratio)

    where both sides are remapped under the same ``f_divergence_type`` /
    ``alpha``. The resulting ``delta`` is the generalised drop-in replacement for
    the plain ``chosen_logratio - rejected_logratio`` term consumed by the DPO
    variants. Under ``reverse_kl`` this reduces exactly to that plain difference.

    The function is elementwise and works on batched ``(B,)`` inputs as well as
    0-dim scalars, making it safe to call under
    ``torch.func.vmap(torch.func.grad(...))``.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response. Shape
            ``(B,)`` or ``()``.
        rejected_logratio: Per-example log-ratio for the rejected response. Same
            shape as ``chosen_logratio``.
        f_divergence_type: Which f-divergence remap to apply to both sides.
            Defaults to ``"reverse_kl"``.
        alpha: The α parameter for ``alpha_divergence`` (must not equal ``1``).
            Ignored by the other divergences. Defaults to ``1.0``.

    Returns:
        Remapped preference-logit tensor of the same shape as the inputs.

    Raises:
        ValueError: Propagated from :func:`f_divergence_remap` for an invalid
            ``alpha_divergence`` ``alpha == 1`` or an unknown
            ``f_divergence_type``.
    """
    chosen = f_divergence_remap(
        chosen_logratio, f_divergence_type=f_divergence_type, alpha=alpha
    )
    rejected = f_divergence_remap(
        rejected_logratio, f_divergence_type=f_divergence_type, alpha=alpha
    )
    return chosen - rejected
