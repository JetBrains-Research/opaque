"""Length-normalised DPO sigmoid loss.

Mathematically identical to the standard DPO sigmoid loss
(``_sigmoid.py``), but the caller is expected to pass log-ratios that have
already been normalised by each completion's token count.  Length
normalisation corrects for the bias that long completions accumulate more
per-token log-probability mass, distorting the pair comparison.

**DP-purity: Tier 1.** Strictly per-example; the length-divisor is
per-example data (applied by the caller before this function is invoked).
NaN-injection contract holds. Vmap-safe.

References:
    Standard DPO sigmoid loss — Rafailov et al., "Direct Preference
    Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS
    2023).  Length normalisation is a common practitioner adaptation; see
    e.g. the ``sigmoid_norm`` entry in TRL ``dpo_trainer.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["sigmoid_norm_loss"]


def sigmoid_norm_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Length-normalised DPO sigmoid per-example loss.

    The formula is identical to the standard DPO sigmoid loss::

        L = -log σ(β·Δ) · (1 - ε) - log σ(-β·Δ) · ε

    where ``Δ = chosen_logratio - rejected_logratio`` and ``ε`` is the
    label-smoothing coefficient.  The length normalisation is the caller's
    responsibility: pass log-ratios that are already divided by the
    respective completion lengths.

    Args:
        chosen_logratio: Per-example scalar log-ratio for the chosen
            completion (already length-normalised by the caller).  May be
            0-dim or ``(B,)``.
        rejected_logratio: Per-example scalar log-ratio for the rejected
            completion (already length-normalised by the caller).  Same
            shape as *chosen_logratio*.
        beta: KL-regularisation temperature (DPO β).
        label_smoothing: Conservative label-smoothing coefficient *ε* in
            ``[0, 0.5)``.  ``0.0`` (default) recovers the unsmoothed DPO
            loss.

    Returns:
        Per-example scalar loss (same shape as inputs).  All operations are
        element-wise; the function is vmap-safe.
    """
    logits = chosen_logratio - rejected_logratio
    return (
        -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
