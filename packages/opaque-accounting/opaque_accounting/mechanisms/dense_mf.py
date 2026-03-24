"""Dense MF mechanism — dense matrix factorization accounting.

Provides privacy accounting for the dense matrix factorization correlated
noise mechanism.  The encoder is a full n x n lower-triangular matrix
optimized for fixed-epoch participation, and the sensitivity is computed
internally.

References:
    - Denisov et al. (2022) https://arxiv.org/abs/2202.08312
    - Choquette-Choo et al. (2022) https://arxiv.org/abs/2211.06530
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    CgfPld,
    DpProcess,
    PmfPld,
)
from opaque_accounting.discretization import _make_native_config


@dataclass(frozen=True, slots=True)
class DenseMf(DpProcess):
    """Dense MF mechanism — dense matrix factorization correlated noise.

    Represents the privacy cost of an entire dense MF training run under
    fixed-epoch participation.
    """

    noise_multiplier: float
    n_steps: int
    epochs: int
    bands: int | None
    equal_norm: bool

    @functools.lru_cache(maxsize=1)
    def _optimized_strategy(self):
        """Optimize the dense strategy matrix and cache it."""
        from opaque.noise.matrix_factorization.dense import (
            optimize,
        )

        return optimize(
            self.n_steps,
            epochs=self.epochs,
            bands=self.bands,
            equal_norm=self.equal_norm,
        )

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under fixed-epoch participation."""
        from opaque.noise.matrix_factorization.sensitivity import (
            fixed_epoch_sensitivity,
        )

        C = self._optimized_strategy()
        return fixed_epoch_sensitivity(C, self.epochs)

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return CgfPld(_native.cgf_mf_gaussian_pld(
            self.noise_multiplier, self.sensitivity()
        ))

    def pmf(self, **kwargs: object) -> PmfPld:
        return PmfPld(_native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity(),
            _make_native_config(**kwargs),
        ))


def dense_mf(
    noise_multiplier: float,
    n_steps: int,
    *,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
) -> DenseMf:
    """Dense MF mechanism — dense matrix factorization correlated noise.

    Args:
        noise_multiplier: Raw noise standard deviation sigma. Must be positive.
        n_steps: Number of training iterations. Must be >= 1.
        epochs: Number of epochs (for fixed-epoch participation). Must
            divide ``n_steps``.
        bands: Number of bands in the strategy (optional).
        equal_norm: If True, optimize with equal column norm constraint.

    Returns:
        A :class:`DenseMf` process.

    Example::

        proc = acc.dense_mf(noise_multiplier=1.0, n_steps=100, epochs=2)
        eps = proc.pmf().epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if n_steps % epochs != 0:
        raise ValueError(f"epochs={epochs} must divide n_steps={n_steps}")
    if bands is not None and bands < 1:
        raise ValueError(f"bands must be >= 1 or None, got {bands}")
    return DenseMf(noise_multiplier, n_steps, epochs, bands, equal_norm)
