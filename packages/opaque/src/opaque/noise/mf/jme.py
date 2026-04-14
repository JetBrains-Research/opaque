"""JME (Joint Moment Estimation) — math helpers for DP-Adam via MF.

Implements the sensitivity analysis from Kalinin, Upadhyay, Lampert (2025)
"Continual Release Moment Estimation with Differential Privacy"
(arXiv:2502.06597, NeurIPS 2025).

JME is NOT a separate noise mechanism — it is a calibration result that
allows two independent ``mf_noise`` streams (for gradients and squared
gradients) to share a single privacy budget. The key result (Theorem 3.2):
with optimal λ, the joint sensitivity of estimating both moments equals
the first-moment-only sensitivity, so the second moment is "free".

Usage: the training loop creates two ``mf_noise`` calls (one per Adam
moment) and uses these helpers to compute the correct noise scales.
See ``examples/train_dp_ftrl.py --optimizer adam`` for the full pattern.

References:
    - JME: https://arxiv.org/abs/2502.06597
"""

from __future__ import annotations

import math

__all__ = [
    "jme_lambda",
    "jme_joint_sensitivity",
    "jme_second_moment_stddev",
]


# ---------------------------------------------------------------------------
# Constants from the paper (Section 3, Algorithm 1)
# ---------------------------------------------------------------------------

_C_D_1 = 8.0 / (11.0 + 5.0 * math.sqrt(5.0))  # ≈ 0.339, for d=1
_C_D_GE2 = 2.0  # for d ≥ 2


def _c_d(d: int) -> float:
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return _C_D_1 if d == 1 else _C_D_GE2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def jme_lambda(
    c1_sensitivity: float,
    c2_sensitivity: float,
    zeta: float,
    d: int = 2,
) -> float:
    """Optimal JME scaling parameter λ (Algorithm 1).

    Sets λ so that estimating the second moment is "free": the joint
    sensitivity equals the first-moment-only sensitivity.

    ``λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})``

    Args:
        c1_sensitivity: Max column norm of C₁ (first moment strategy).
            For single-participation, this is ``strategy.sensitivity``.
        c2_sensitivity: Max column norm of C₂ (second moment strategy).
        zeta: Per-sample clipping bound (``clip_state.sensitivity``).
        d: Dimension. ``d ≥ 2`` (default) for neural nets; ``d = 1`` for scalars.

    Returns:
        The optimal scaling parameter λ.
    """
    if c1_sensitivity <= 0:
        raise ValueError(f"c1_sensitivity must be positive, got {c1_sensitivity}")
    if c2_sensitivity <= 0:
        raise ValueError(f"c2_sensitivity must be positive, got {c2_sensitivity}")
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    cd = _c_d(d)
    return (c1_sensitivity**2) / (cd * zeta**2 * c2_sensitivity**2)


def jme_joint_sensitivity(c1_sensitivity: float, zeta: float) -> float:
    """Joint sensitivity for both moments (Theorem 3.2).

    ``s = 2ζ · ‖C₁‖_{1→2}``

    This is 2× the first-moment-only sensitivity.  Privacy accounting
    uses this value (via ``noise_multiplier * s``) to cover *both*
    moment streams at no additional cost.

    Args:
        c1_sensitivity: Max column norm of C₁.
        zeta: Per-sample clipping bound.

    Returns:
        The joint sensitivity.
    """
    if c1_sensitivity <= 0:
        raise ValueError(f"c1_sensitivity must be positive, got {c1_sensitivity}")
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    return 2.0 * zeta * c1_sensitivity


def jme_second_moment_stddev(
    first_moment_stddev: float,
    lambda_jme: float,
) -> float:
    """Noise stddev for the second moment stream.

    The second moment noise is scaled by ``λ^{-1/2}`` relative to the
    first moment noise.

    Args:
        first_moment_stddev: Noise stddev used for the first moment
            (gradient) stream, i.e. ``noise_multiplier * joint_sensitivity``.
        lambda_jme: The λ from :func:`jme_lambda`.

    Returns:
        The stddev for the second moment (squared-gradient) noise stream.
    """
    if lambda_jme <= 0:
        raise ValueError(f"lambda_jme must be positive, got {lambda_jme}")
    return first_moment_stddev / math.sqrt(lambda_jme)
