"""LR-schedule-aware Toeplitz square root factorization (2511.17994).

For exponential LR decay :math:`\\chi_t = \\beta^{(t-1)/(n-1)}`, the paper
proposes :math:`C_\\alpha = (A_\\chi^{\\text{Toep}})^{1/2}` with closed-form
coefficients :math:`c_j = \\alpha^j r_j` where :math:`\\alpha = \\beta^{1/(n-1)}`
and :math:`r_j = |(-1/2 \\text{ choose } j)|`.

For multi-epoch, band :math:`C_\\alpha` to width :math:`p` (same as BSR banding).
Sensitivity and Gram use the standard Toeplitz min-sep machinery.

References:
    - Kalinin & Andersson (2025), arXiv:2511.17994
    - Related: BSR (Kalinin & Lampert, arXiv:2405.13763) — same coefficient family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from opaque_accounting import opaque_accounting as _native

from ._sensitivity import minsep_true_max_participations
from ._streaming_matrix import StreamingMatrix
from ._toeplitz import (
    inverse_as_streaming_matrix,
    sensitivity_squared as _toeplitz_col_norm_sq,
    minsep_sensitivity_squared as _toeplitz_minsep_sensitivity_squared,
)
from .bsr import _r_sequence

__all__ = ["LrAwareStrategy", "lr_aware_strategy"]


def _lr_aware_coefficients(bandwidth: int, alpha: float) -> list[float]:
    """Closed-form :math:`C_\\alpha` coefficients: :math:`c_j = \\alpha^j r_j`.

    Equivalent to BSR with :math:`(\\alpha, \\beta=0)`.
    """
    if bandwidth < 1:
        raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must be in (0, 1) for exponential decay, got {alpha}"
        )
    r = _r_sequence(bandwidth)
    return [alpha**j * r[j] for j in range(bandwidth)]


@dataclass(frozen=True, slots=True)
class LrAwareStrategy:
    """LR-schedule-aware Toeplitz square root strategy (arXiv:2511.17994)."""

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: tuple[float, ...] | None = None
    _streaming_matrix: StreamingMatrix | None = None
    _max_column_norm: float = 0.0
    _bandwidth: int = 1
    _n_steps: int = 1
    _min_sep: int = 1
    _max_participations: int | None = 1
    _alpha: float = 0.99
    _lr_decay_beta: float = 0.5


def lr_aware_strategy(
    bandwidth: int,
    n_steps: int,
    min_sep: int,
    max_participations: int | None = 1,
    *,
    lr_decay_beta: float,
) -> LrAwareStrategy:
    """Create an LR-schedule-aware strategy for exponential LR decay.

    For :math:`\\chi_t = \\beta^{(t-1)/(n-1)}` (learning rate decays from 1 to
    ``lr_decay_beta`` over ``n_steps``), computes the Toeplitz square root
    :math:`C_\\alpha` with :math:`\\alpha = \\beta^{1/(n-1)}`.

    Args:
        bandwidth: Bandwidth p (>= 1). Coefficients beyond p are zero.
        n_steps: Total training steps.
        min_sep: Minimum separation between participations (steps per epoch).
        max_participations: Maximum participations per user (epochs).
        lr_decay_beta: Final-to-initial LR ratio in (0, 1). The LR schedule
            is :math:`\\eta_t = \\eta \\cdot \\beta^{(t-1)/(n-1)}`.

    Returns:
        A :class:`LrAwareStrategy` with Gram matrix for BnB accounting.

    Raises:
        ValueError: If ``lr_decay_beta`` is not in (0, 1).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if min_sep < 1:
        raise ValueError(f"min_sep must be >= 1, got {min_sep}")
    if not (0.0 < lr_decay_beta < 1.0):
        raise ValueError(
            f"lr_decay_beta must be in (0, 1), got {lr_decay_beta}. "
            "Use 0.5 for halving, 0.25 for quarter, etc."
        )
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 for exponential decay, got {n_steps}")

    alpha = lr_decay_beta ** (1.0 / (n_steps - 1))

    band_coefs = _lr_aware_coefficients(bandwidth, alpha)
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

    gram = _native.toeplitz_gram_matrix(
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

    return LrAwareStrategy(
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
        _lr_decay_beta=lr_decay_beta,
    )
