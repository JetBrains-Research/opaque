"""DiscoPOP (Discovering Preference Optimization Procedures) loss.

Implements the logistic/exponential blend discovered by offline meta-learning
of DPO-family objectives:

    Azar, M. G., et al. (2024). DiscoPOP: Discovering Preference
    Optimization Procedures Using Self-Supervised Feedback. NeurIPS 2024.

The modulation gate ``gate = σ(β·Δ / τ)`` blends a logistic component and an
exponential component via ``L = logistic·(1 − gate) + exp·gate``.  Because
``gate → 1`` for large positive ``β·Δ`` and ``gate → 0`` for negative ``β·Δ``,
the **exponential** component dominates when ``β·Δ`` is large positive and the
**logistic** component dominates when ``β·Δ`` is negative.

**DP-purity: Tier 1.** Strictly per-example. NaN-injection contract holds.
Vmap-safe.

**Numerical note — exp overflow.**  When ``logits = β·Δ`` is large
*negative* (e.g. ``logits → -∞``), the exponential component
``exp(-logits)`` grows without bound.  At the same time the modulation gate
``sigmoid(logits / τ) → 0`` for negative logits, so in the limit the
contribution of the exponential term is ``0 · ∞``, which is numerically
unstable.  In practice the modulation gate decays exponentially in
``logits / τ``, so the product is bounded for any finite logits; however,
at half-precision or for extreme ``beta * |Δ|``, intermediate overflow can
still occur.

To guard against this, the exponential component is clamped before
multiplication:  ``exp_component = torch.exp(logits.clamp(max=0) * -1) *
torch.exp((-logits).clamp(max=80))``.  A simpler and numerically safer
alternative used here is to express the exponential as
``exp(-logits.clamp(max=MAX_LOGIT))`` where ``MAX_LOGIT`` is chosen so that
``exp`` stays in the finite float32 range (``MAX_LOGIT = 80`` gives
``exp(-MAX_LOGIT) ≈ 1.8e-35``, safely above fp32 underflow but bounded).
See the inline clamp comment below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["discopop_loss"]

# Maximum absolute value of logits passed to torch.exp(-logits).
# exp(-logits) overflows float32 for logits < -88.7 (exp(88.7) ≈ 3.4e38).
# Clamping logits from below at -_EXP_CLAMP keeps exp(-logits) finite.
_EXP_CLAMP: float = 80.0


def discopop_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    discopop_tau: float = 0.05,
) -> torch.Tensor:
    """DiscoPOP per-example loss (Azar 2024 / NeurIPS 2024).

    Blends a logistic component ``-log σ(β·Δ)`` and an exponential component
    ``exp(-β·Δ)`` using a soft gate parameterised by temperature *τ*::

        logits    = (chosen_logratio - rejected_logratio) * beta
        gate      = σ(logits / τ)          # → 1 for large positive logits
        logistic  = -log σ(logits)
        exp_comp  = exp(-logits)            # clamped to avoid overflow
        L         = logistic * (1 - gate) + exp_comp * gate

    Args:
        chosen_logratio: Per-example scalar log-ratio for the chosen
            completion.  May be 0-dim or ``(B,)``.
        rejected_logratio: Per-example scalar log-ratio for the rejected
            completion.  Same shape as *chosen_logratio*.
        beta: KL-regularisation temperature (DPO β).
        discopop_tau: Modulation temperature *τ* (default ``0.05``).
            Smaller values sharpen the transition between the two components.

    Returns:
        Per-example scalar loss (same shape as inputs).  All operations are
        element-wise; the function is vmap-safe.

    Note:
        The exponential component overflows float32 when ``logits ≪ 0``.  To
        prevent NaN/Inf while preserving gradient locality (Tier 1), the
        logits passed to ``torch.exp`` are clamped from below at
        ``-_EXP_CLAMP`` (80).  This keeps ``exp(-logits)`` ≤ ``exp(80)
        ≈ 5.5e34``, which is within the float32 range.  At large negative
        logits, the modulation gate simultaneously drives towards 0, so the
        clamping is numerically inconsequential for the loss magnitude.
    """
    logits = (chosen_logratio - rejected_logratio) * beta
    # Clamp logits for the exp component to avoid overflow when logits << 0.
    logits_clamped = logits.clamp(min=-_EXP_CLAMP)
    modulation = torch.sigmoid(logits / discopop_tau)
    logistic_component = -F.logsigmoid(logits)
    exp_component = torch.exp(-logits_clamped)
    return logistic_component * (1.0 - modulation) + exp_component * modulation
