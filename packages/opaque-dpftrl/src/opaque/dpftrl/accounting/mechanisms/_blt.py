"""BLT mechanism with pre-computed Gram matrix.

References:
    - BLT: Choquette-Choo et al. (2024) https://arxiv.org/abs/2404.16706
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.dpftrl.accounting.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Blt(MfGaussian):
    """BLT mechanism with pre-computed Gram matrix.

    The ``gram_matrix`` (flattened row-major) is needed by
    :func:`~opaque.dpftrl.accounting.amplification._balls_in_bins.balls_in_bins`.
    Without it, provides unamplified BLT accounting.
    """

    gram_matrix: tuple[float, ...] = ()


def blt(
    noise_multiplier: float,
    sensitivity: float,
    gram_matrix: tuple[float, ...] = (),
) -> Blt:
    """BLT mechanism — unamplified or with Gram matrix for BnB.

    Without ``gram_matrix``, provides unamplified BLT accounting.
    With ``gram_matrix``, can be wrapped in :func:`balls_in_bins`.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the BLT strategy.
        gram_matrix: Pre-computed Gram matrix (flattened row-major).
            Empty tuple for unamplified accounting.

    Returns:
        A :class:`Blt` process.

    Example::

        # Unamplified
        proc = ftrl_acc.blt(1.0, sensitivity=s.sensitivity)

        # With BnB amplification
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.blt(1.0, sensitivity=s.sensitivity,
                         gram_matrix=s.gram_matrix),
            num_bins=1953, num_epochs=8,
        )
    """
    return Blt(noise_multiplier, sensitivity, gram_matrix)
