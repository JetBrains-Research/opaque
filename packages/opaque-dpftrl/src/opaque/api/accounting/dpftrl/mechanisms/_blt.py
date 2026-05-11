"""BLT mechanism with pre-computed Gram matrix.

References:
    - BLT: Choquette-Choo et al. (2024) https://arxiv.org/abs/2404.16706
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class Blt(MfGaussian):
    """BLT mechanism with pre-computed Gram matrix.

    Construct via :meth:`opaque.dpftrl.noise.BltStrategy.as_mechanism` rather
    than calling this dataclass directly — the strategy supplies all the
    structural fields the BnB amplification (and :meth:`with_horizon`) needs.
    """

    gram_matrix: tuple[float, ...]
    coefficients: tuple[float, ...]
    min_sep: int
    max_participations: int | None

    def with_horizon(
        self, n_steps: int, max_participations: int | None
    ) -> "Blt":
        """Return a copy with Gram regenerated for a shorter horizon."""
        from opaque.api.accounting.core import _native

        new_gram = tuple(
            _native.toeplitz_gram_matrix(
                list(self.coefficients),
                n_steps,
                self.min_sep,
                max_participations,
                True,
            )
        )
        return dataclasses.replace(self, gram_matrix=new_gram)
