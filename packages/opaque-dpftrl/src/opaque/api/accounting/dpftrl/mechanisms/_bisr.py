"""BISR mechanism with pre-computed Gram matrix.

References:
    - BISR: Kalinin et al. (ICLR 2026) https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Bisr(MfGaussian):
    """BISR mechanism with pre-computed Gram matrix.

    The ``gram_matrix`` (flattened row-major) is needed by
    :func:`~opaque.dpftrl.accounting.amplification._balls_in_bins.balls_in_bins`.
    """

    gram_matrix: tuple[float, ...] = ()


def bisr(
    noise_multiplier: float,
    sensitivity: float,
    gram_matrix: tuple[float, ...] = (),
) -> Bisr:
    """BISR mechanism with Gram matrix for BnB amplification.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the BISR strategy.
        gram_matrix: Pre-computed Gram matrix (flattened row-major).

    Returns:
        A :class:`Bisr` process.

    Example::

        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.bisr(1.0, sensitivity=s.sensitivity,
                          gram_matrix=s.gram_matrix),
            num_bins=1953, n_steps=15624,
        )
    """
    return Bisr(noise_multiplier, sensitivity, gram_matrix)
