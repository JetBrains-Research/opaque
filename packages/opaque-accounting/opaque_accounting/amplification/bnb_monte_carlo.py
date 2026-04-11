"""Monte Carlo BnB accounting for matrix mechanisms.

Implements the paper-correct BnB accounting from Choquette-Choo et al.
(2024) "Near Exact Privacy Amplification for Matrix Mechanisms"
(arxiv:2410.06266).

Uses the Gram matrix of the dominating pair mixture means to sample the
privacy loss distribution via Monte Carlo, producing a standard PLD.

This replaces the incorrect Poisson-composition approach for correlated
noise mechanisms (e.g. DP-λCGD).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.discretization import get_discretization


@dataclass(frozen=True, slots=True)
class BnBMonteCarlo(DpProcess):
    """BnB-amplified mechanism with Monte Carlo PLD estimation.

    Samples from the BnB dominating pair (Lemma 3.2 of arxiv:2410.06266),
    then discretizes onto the standard PLD grid.

    This represents the **total** privacy cost for multi-epoch training.
    Do NOT compose further with ``* num_epochs``::

        # CORRECT usage:
        training = acc.balls_in_bins(
            acc.lambda_cgd(nm, lambda_=0.9, n_steps=SPE * num_epochs,
                           min_sep=SPE, max_participations=num_epochs),
            num_bins=SPE,
        )
        eps = training.epsilon_at(1e-5)  # total cost, no * needed

    The Gram matrix G captures the inner products of the dominating pair
    mixture means: G_{ij} = ⟨m_i, m_j⟩ where m_i = Σ_epoch C[:,b·epoch+i].
    """

    gram: tuple[float, ...]
    num_bins: int
    noise_multiplier: float
    num_samples: int = 1_000_000
    seed: int = 42

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        return _native.bnb_mc_pld(
            list(self.gram),
            self.num_bins,
            self.noise_multiplier,
            self.num_samples,
            self.seed,
            config.to_native(),
        )

    def __mul__(self, count: int) -> DpProcess:
        raise TypeError(
            "BnBMonteCarlo represents the full multi-epoch training cost. "
            "Do NOT multiply by num_epochs — the BnB analysis already "
            "covers all epochs. Pass n_steps=SPE*num_epochs to lambda_cgd()."
        )

    def __rmul__(self, count: int) -> DpProcess:
        return self.__mul__(count)
