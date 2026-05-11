"""BISR mechanism with pre-computed Gram matrix.

References:
    - BISR: Kalinin et al. (ICLR 2026) https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Bisr(MfGaussian):
    """BISR mechanism with pre-computed Gram matrix.

    Construct via :meth:`opaque.dpftrl.noise.BisrStrategy.as_mechanism`.
    ``inv_coefficients`` is the bandwidth-length C^{-1} band sequence.
    """

    gram_matrix: tuple[float, ...]
    inv_coefficients: tuple[float, ...]
    min_sep: int
    max_participations: int | None
    normalized: bool

    def with_horizon(self, n_steps: int, max_participations: int | None) -> "Bisr":
        """Return a copy with Gram regenerated for a shorter horizon."""
        from opaque.api.accounting.core import _native

        new_gram = tuple(
            _native.bisr_gram_matrix(
                list(self.inv_coefficients),
                n_steps,
                self.min_sep,
                max_participations,
                self.normalized,
            )
        )
        return dataclasses.replace(self, gram_matrix=new_gram)
