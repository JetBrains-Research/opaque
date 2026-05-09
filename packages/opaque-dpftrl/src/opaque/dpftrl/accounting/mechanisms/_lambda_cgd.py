"""DP-λCGD mechanism with pre-computed Gram matrix.

References:
    - DP-λCGD: Kalinin et al. (2026) https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.dpftrl.accounting.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class LambdaCgd(MfGaussian):
    """DP-λCGD mechanism with pre-computed Gram matrix.

    The ``gram_matrix`` (flattened row-major) is needed by
    :func:`~opaque.dpftrl.accounting.amplification._balls_in_bins.balls_in_bins`.
    """

    gram_matrix: tuple[float, ...] = ()


def lambda_cgd(
    noise_multiplier: float,
    sensitivity: float,
    gram_matrix: tuple[float, ...] = (),
) -> LambdaCgd:
    """DP-λCGD mechanism with Gram matrix for BnB amplification.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the λCGD strategy.
        gram_matrix: Pre-computed Gram matrix (flattened row-major).

    Returns:
        A :class:`LambdaCgd` process.

    Example::

        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(1.0, sensitivity=s.sensitivity,
                                gram_matrix=s.gram_matrix),
            num_bins=1953, n_steps=15624,
        )
    """
    return LambdaCgd(noise_multiplier, sensitivity, gram_matrix)
