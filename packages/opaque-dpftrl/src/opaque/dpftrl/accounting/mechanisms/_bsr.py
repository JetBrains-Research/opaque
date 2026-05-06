"""BSR mechanism with pre-computed Gram matrix (Kalinin & Lampert, NeurIPS 2024).

References:
    - BSR: https://arxiv.org/abs/2405.13763
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.dpftrl.accounting.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Bsr(MfGaussian):
    """BSR mechanism with pre-computed Gram matrix for Balls-in-Bins."""

    gram_matrix: tuple[float, ...] = ()


def bsr(
    noise_multiplier: float,
    sensitivity: float,
    gram_matrix: tuple[float, ...] = (),
) -> Bsr:
    """BSR matrix-factorization mechanism.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the BSR strategy.
        gram_matrix: Pre-computed Gram matrix (flattened) for ``balls_in_bins``.

    Returns:
        A :class:`Bsr` process.
    """
    return Bsr(noise_multiplier, sensitivity, gram_matrix)
