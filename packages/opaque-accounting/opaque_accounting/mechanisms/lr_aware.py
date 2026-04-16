"""LR-aware mechanism with pre-computed Gram matrix (Kalinin & Andersson, 2025).

References:
    - Kalinin & Andersson (2025), arXiv:2511.17994
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque_accounting.mechanisms.mf_gaussian import MfGaussian


@dataclass(frozen=True)
class LrAware(MfGaussian):
    """LR-schedule-aware mechanism with pre-computed Gram matrix for BnB."""

    gram_matrix: tuple[float, ...] = ()


def lr_aware(
    noise_multiplier: float,
    sensitivity: float,
    gram_matrix: tuple[float, ...] = (),
) -> LrAware:
    """LR-schedule-aware matrix-factorization mechanism.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the LR-aware strategy.
        gram_matrix: Pre-computed Gram matrix (flattened) for ``balls_in_bins``.

    Returns:
        A :class:`LrAware` process.
    """
    return LrAware(noise_multiplier, sensitivity, gram_matrix)
