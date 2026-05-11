"""BSR mechanism with pre-computed Gram matrix (Kalinin & Lampert, NeurIPS 2024).

References:
    - BSR: https://arxiv.org/abs/2405.13763
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Bsr(MfGaussian):
    """BSR mechanism with pre-computed Gram matrix for Balls-in-Bins.

    Construct via :meth:`opaque.dpftrl.noise.BsrStrategy.as_mechanism`.
    ``coefficients`` is the bandwidth-length band sequence (not n_steps-padded).
    """

    gram_matrix: tuple[float, ...]
    coefficients: tuple[float, ...]
    min_sep: int
    max_participations: int | None

    def with_horizon(
        self, n_steps: int, max_participations: int | None
    ) -> "Bsr":
        """Return a copy with Gram regenerated for a shorter horizon."""
        from opaque.api.accounting.core import _native

        new_gram = tuple(
            _native.toeplitz_gram_matrix(
                list(self.coefficients),
                n_steps,
                self.min_sep,
                max_participations,
                False,
            )
        )
        return dataclasses.replace(self, gram_matrix=new_gram)
