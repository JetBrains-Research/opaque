"""DP-λCGD mechanism with pre-computed Gram matrix.

References:
    - DP-λCGD: Kalinin et al. (2026) https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class LambdaCgd(MfGaussian):
    """DP-λCGD mechanism with pre-computed Gram matrix.

    Construct via :meth:`opaque.dpftrl.noise.LambdaCgdStrategy.as_mechanism`.
    """

    gram_matrix: tuple[float, ...]
    lambda_: float
    min_sep: int
    max_participations: int | None
    normalized: bool

    def with_horizon(
        self, n_steps: int, max_participations: int | None
    ) -> "LambdaCgd":
        """Return a copy with Gram regenerated for a shorter horizon."""
        from opaque.api.accounting.core import _native

        new_gram = tuple(
            _native.lambda_cgd_gram_matrix(
                self.lambda_,
                n_steps,
                self.min_sep,
                max_participations,
                self.normalized,
            )
        )
        return dataclasses.replace(self, gram_matrix=new_gram)
