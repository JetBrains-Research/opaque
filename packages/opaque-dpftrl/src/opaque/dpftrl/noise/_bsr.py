"""BSR strategy — banded square root MF (Kalinin & Lampert, NeurIPS 2024).

Closed-form lower-triangular Toeplitz coefficients for the paper workload
:math:`A_{\\alpha,\\beta}` (multiplicative decay :math:`\\alpha`, Polyak momentum
:math:`\\beta`). These are **not** the same names as PyTorch ``weight_decay`` or
generic ``momentum`` on other strategies—here ``alpha`` and ``beta`` match the paper.

No numerical optimization.

Use ``mf_noise(bsr_strategy(...), ...)`` to create the noise function.

References:
    - BSR: https://arxiv.org/abs/2405.13763 (Theorem 1, banded :math:`C^{|p|}`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._sensitivity import minsep_true_max_participations
from ._streaming_matrix import StreamingMatrix
from ._toeplitz import (
    inverse_as_streaming_matrix,
    sensitivity_squared as _toeplitz_col_norm_sq,
    minsep_sensitivity_squared as _toeplitz_minsep_sensitivity_squared,
)


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


__all__ = ["BsrStrategy", "bsr_strategy"]


def _r_sequence(length: int) -> list[float]:
    """Binomial-type coefficients r_k = |(-1/2 choose k)| (paper Theorem 1)."""
    if length < 1:
        return []
    r = [1.0]
    for k in range(1, length):
        r.append(r[-1] * (k - 0.5) / k)
    return r


def _bsr_coefficients(bandwidth: int, alpha: float, beta: float) -> list[float]:
    """First-column coefficients of :math:`C^{|p|}_{\\alpha,\\beta}` (Theorem 1).

    :math:`c_0 = 1`, :math:`c_j = \\sum_{i=0}^{j} \\alpha^{j-i} r_{j-i} r_i \\beta^i`
    for :math:`j = 1,\\ldots,p-1`.
    """
    if bandwidth < 1:
        raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
    r = _r_sequence(bandwidth)
    c = [1.0]
    for j in range(1, bandwidth):
        s = 0.0
        for i in range(j + 1):
            s += (alpha ** (j - i)) * r[j - i] * r[i] * (beta**i)
        c.append(s)
    return c


def _validate_bsr_hyperparams(
    bandwidth: int,
    alpha: float,
    beta: float,
) -> None:
    """Raise ValueError if hyperparameters are outside the supported v1 regime."""
    if bandwidth < 1:
        raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
    if not (0.0 <= beta < 1.0):
        raise ValueError(
            f"BSR v1 requires β in [0, 1), got {beta}. "
            "Use band_mf_strategy for other workloads."
        )
    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            f"BSR v1 requires α in (0, 1] (paper), got {alpha}. "
            "Use band_mf_strategy for other workloads."
        )
    if alpha <= beta:
        raise ValueError(
            f"BSR v1 requires α > β (paper regime); got α={alpha}, β={beta}. "
            "Reduce β or increase α, or use band_mf_strategy."
        )
    coefs = _bsr_coefficients(bandwidth, alpha, beta)
    for i in range(1, len(coefs)):
        if coefs[i] > coefs[i - 1] + 1e-12:
            raise ValueError(
                "BSR coefficients are not non-increasing for these hyperparameters; "
                "this configuration is not supported by toeplitz_minsep_sensitivity. "
                "Try different α, β, or bandwidth, or use band_mf_strategy."
            )
        if coefs[i] < -1e-12:
            raise ValueError(
                "BSR produced a negative coefficient; unsupported for this Rust path. "
                "Use band_mf_strategy."
            )


@dataclass(frozen=True, slots=True)
class BsrStrategy:
    """BSR (banded square root) strategy; workload parameters :math:`\\alpha,\\beta`."""

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: tuple[float, ...] | None = None
    _streaming_matrix: StreamingMatrix | None = None
    _max_column_norm: float = 0.0
    _bandwidth: int = 1
    _n_steps: int = 1
    _min_sep: int = 1
    _max_participations: int | None = 1
    _alpha: float = 1.0
    _beta: float = 0.0


def bsr_strategy(
    bandwidth: int,
    n_steps: int,
    min_sep: int,
    max_participations: int | None = 1,
    *,
    alpha: float,
    beta: float,
) -> BsrStrategy:
    """Create a BSR strategy with closed-form Toeplitz coefficients.

    Coefficients follow Theorem 1 of arXiv:2405.13763 (banded square root of
    :math:`A_{\\alpha,\\beta}`). Sensitivity uses Theorem 2 via
    ``toeplitz_minsep_sensitivity_squared``. Gram matrix is for Balls-in-Bins
    accounting (same as BLT/BISR Toeplitz Gram).

    Args:
        bandwidth: Bandwidth p (>= 1). Only c_0..c_{p-1} are non-zero.
        n_steps: Total training steps (matrix dimension).
        min_sep: Minimum separation between participations (steps per epoch).
        max_participations: Maximum participations per user (epochs).
        alpha: Paper workload decay :math:`\\alpha \\in (0, 1]`. **Not** AdamW
            ``weight_decay`` in PyTorch units.
        beta: Paper Polyak momentum :math:`\\beta \\in [0, 1)`. For training scripts,
            bind from SGD ``momentum`` or Adam :math:`\\beta_1` (first-moment EMA) so
            the noise workload matches the optimizer you analyze.

    Returns:
        A :class:`BsrStrategy` with Gram matrix for BnB.

    Raises:
        ValueError: If hyperparameters are outside the supported closed-form regime.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if min_sep < 1:
        raise ValueError(f"min_sep must be >= 1, got {min_sep}")

    _validate_bsr_hyperparams(bandwidth, alpha, beta)

    band_coefs = _bsr_coefficients(bandwidth, alpha, beta)
    coef_tensor = torch.zeros(n_steps, dtype=torch.float64)
    copy_len = min(bandwidth, n_steps)
    coef_tensor[:copy_len] = torch.tensor(band_coefs[:copy_len], dtype=torch.float64)

    max_col_sq = _toeplitz_col_norm_sq(coef_tensor, n_steps)
    max_column_norm = float(max_col_sq.sqrt())

    k = minsep_true_max_participations(
        n=n_steps, min_sep=min_sep, max_participations=max_participations
    )
    if k == 1:
        sensitivity = max_column_norm
    else:
        sens_sq = _toeplitz_minsep_sensitivity_squared(
            strategy_coef=coef_tensor,
            min_sep=min_sep,
            max_participations=max_participations,
            skip_checks=True,
        )
        sensitivity = float(sens_sq.sqrt())

    coefficients = tuple(coef_tensor.tolist())

    gram = _native().toeplitz_gram_matrix(
        band_coefs,
        n_steps,
        min_sep,
        max_participations,
        False,
    )
    gram_matrix = tuple(gram)

    streaming = inverse_as_streaming_matrix(
        coef_tensor[:copy_len].clone(),
        column_normalize_for_n=None,
    )

    return BsrStrategy(
        sensitivity=sensitivity,
        coefficients=coefficients,
        gram_matrix=gram_matrix,
        _streaming_matrix=streaming,
        _max_column_norm=max_column_norm,
        _bandwidth=bandwidth,
        _n_steps=n_steps,
        _min_sep=min_sep,
        _max_participations=max_participations,
        _alpha=alpha,
        _beta=beta,
    )
