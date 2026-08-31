"""Balls-in-Bins amplification for DP-FTRL — **total** multi-epoch cost.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins.  The bin assignment
is fixed once at sampler init and reused across all ``n_steps //
num_bins`` epochs, so each example stays in its bin — required for the
dominating-pair analysis.

The Choquette-Choo et al. (2024) dominating pair (Lemma 3.2) is::

    P = (1/b) Σ_{i=1}^{b} N(m_i, σ²I)        m_i = Σ_{j=0}^{E-1} |C|[:, b·j + i]
    Q = N(0, σ²I)

where ``E = n_steps // num_bins`` is the per-bin participation count.

After Gram-matrix reduction (``G[i,j] = m_i · m_j``) the privacy loss only
depends on ``G``, ``num_bins`` and ``σ``.  Both dispatch paths feed this
construction:

- **Correlated-noise** (matrix-factorisation): MfGaussian wrapping
  ``BltStrategy`` / ``BsrStrategy`` / ``BisrStrategy`` /
  ``LambdaCgdStrategy`` — pass the strategy's pre-computed Gram matrix.
- **MF identity** (uncorrelated noise): MfGaussian wrapping
  ``IdentityStrategy`` (encoder ``C = I``) gives orthogonal ``m_i`` with
  ``‖m_i‖² = E``, i.e. ``G = E · I_b`` (diagonal).  This feeds the same
  Lemma 3.2 dominating pair through a specialised MC primitive.

The returned process represents the **total** privacy cost across
all ``n_steps`` rounds.  Do NOT compose further externally.

References:
    - Chua et al. (2025), "Balls-and-Bins Sampling for DP-SGD":
      https://arxiv.org/abs/2412.16802
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core._pld_cache import horizon_pld_cache
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._bisr import BisrStrategy
from opaque.api.dpftrl.noise._blt import BltStrategy
from opaque.api.dpftrl.noise._bsr import BsrStrategy
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy
from opaque.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opaque.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import Pld

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = MfGaussian
_MIN_NUM_BINS = 2

#: Strategy types whose Gram is needed at PLD time (the "correlated MF" set).
_CorrelatedStrategies = (BltStrategy, BsrStrategy, BisrStrategy, LambdaCgdStrategy)


@dataclass(frozen=True, slots=True)
class BallsInBins(DpHorizonProcess):
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    The returned PLD covers all ``n_steps`` training rounds (= ``num_bins``
    bins × ``n_steps // num_bins`` epochs).  Do NOT compose externally.

    For ``IdentityStrategy`` inner, the dispatch uses the deterministic
    random-allocation transform: with ``C = I`` the Gram is exactly
    ``num_epochs · I_b`` and the dominating pair collapses onto
    1-out-of-``num_bins`` random allocation at ``σ/√num_epochs``.

    Example (DP-λCGD)::

        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(nm, strategy),
            num_bins=steps_per_epoch,
            n_steps=steps_per_epoch * num_epochs,
        )
        eps = training.epsilon_at(1e-5)
    """

    inner: _Inner
    num_bins: int
    n_steps: int

    def __post_init__(self) -> None:
        if self.num_bins < _MIN_NUM_BINS:
            raise ConfigurationError(
                *(f"num_bins must be >= 2 for BnB amplification, got {self.num_bins}",)
            )
        if self.n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {self.n_steps}",))
        if self.n_steps % self.num_bins != 0:
            raise ConfigurationError(
                *(
                    f"n_steps ({self.n_steps}) must be a positive multiple of "
                    f"num_bins ({self.num_bins}); BnB analysis assumes integer epochs.",
                )
            )

    @property
    def num_epochs(self) -> int:
        """Per-bin participation count: ``n_steps // num_bins``."""
        return self.n_steps // self.num_bins

    @property
    def min_sep(self) -> int:
        # Each example participates once per epoch; consecutive participations
        # are exactly ``num_bins`` rounds apart.
        return self.num_bins

    @property
    def max_participations(self) -> int:
        # Once per epoch ⇒ ``num_epochs`` total per example.
        return self.num_epochs

    @property
    def atomic_unit(self) -> int:
        # One full epoch covers ``num_bins`` rounds; the BnB dominating-pair
        # analysis is defined at epoch boundaries.  ``per_step(self) * K``
        # rounds K up to the next epoch.
        return self.num_bins

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        prefix_steps = self.n_steps if n_steps is None else n_steps
        if isinstance(self.inner.strategy, IdentityStrategy):
            prefix_steps = min(
                -(-prefix_steps // self.num_bins) * self.num_bins,
                self.n_steps,
            )
        else:
            # Correlated inners share one full-horizon PLD for every prefix.
            prefix_steps = self.n_steps
        return (
            "BallsInBins",
            self.inner.noise_multiplier,
            self.num_bins,
            self.n_steps,
            strategy_cache_key(self.inner.strategy, prefix_steps),
            prefix_steps,
        )

    @horizon_pld_cache(maxsize=8)
    def pld_at(
        self,
        n_steps: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        """Return the BnB PLD for a prefix of ``n_steps`` rounds.

        ``n_steps`` is rounded up to the next epoch (a multiple of
        ``num_bins``, capped at ``self.n_steps``).  ``IdentityStrategy``
        yields an exact per-epoch prefix via the random-allocation PLD
        transform.  Every other inner charges the full-horizon Monte Carlo
        bound for any nonzero prefix.
        """
        from opaque.api.accounting.core.discretization import get_discretization

        if n_steps <= 0 or n_steps > self.n_steps:
            raise ConfigurationError(
                *(f"n_steps ({n_steps}) must be in [1, {self.n_steps}]",)
            )
        rounded = min(-(-n_steps // self.num_bins) * self.num_bins, self.n_steps)
        num_epochs_K = rounded // self.num_bins

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        native_cfg = config.to_native()

        s = self.inner.strategy

        if not isinstance(s, IdentityStrategy) and rounded < self.n_steps:
            return self.pld_at(
                self.n_steps,
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                seed=seed,
                mc_resolution=mc_resolution,
                mc_failure_probability=mc_failure_probability,
            )

        # Identity (C = I) collapses onto random allocation, exactly.
        #
        # The mixture means ``m_i = Σ_j |C|[:, b·j+i]`` are then indicators of
        # disjoint step sets, so they are orthogonal with equal norm ``√E`` and
        # the gram is ``G = E·I_b``.  Projecting onto their span and rescaling
        # by ``1/√E`` turns the Lemma 3.2 pair into
        # ``P = (1/b)Σ N(e_i, σ_eff²I_b)``, ``Q = N(0, σ_eff²I_b)`` with
        # ``σ_eff = σ/√E`` — precisely the 1-out-of-b random allocation pair.
        #
        # So this path uses the deterministic PLD transform rather than Monte
        # Carlo: no 1/δ sample cost, reproducible across thread counts, and
        # tighter.  The sampler is unchanged — bins stay fixed across epochs.
        if isinstance(s, IdentityStrategy):
            if self.inner.noise_multiplier == 0:
                return _native.non_private_pld(native_cfg)
            sigma_eff = float(self.inner.noise_multiplier) / math.sqrt(num_epochs_K)
            return _native.random_allocation_gaussian_pld(
                sigma_eff,
                self.num_bins,
                1,
                native_cfg,
            )

        gram = s.gram_matrix(
            n_steps=self.n_steps,
            min_sep=self.min_sep,
            max_participations=self.max_participations,
        )
        config.warn_if_large_mc()
        return _native.bnb_mc_pld(
            list(gram),
            self.num_bins,
            self.inner.noise_multiplier,
            native_cfg,
        )

    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        """Return the full-horizon PLD, confidence-bounded for MC strategies."""
        return self.pld_at(
            self.n_steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )


def balls_in_bins(
    inner: _Inner,
    *,
    num_bins: int,
    n_steps: int,
) -> BallsInBins:
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    Each epoch, the dataset is partitioned into ``num_bins`` bins (assignment
    fixed at sampler init and reused across epochs).  Every example
    participates exactly once per epoch.  The total round count is
    ``n_steps``; per-bin participation count is ``n_steps // num_bins`` and
    must divide evenly.

    Args:
        inner: ``mf_gaussian(nm, strategy)`` where ``strategy`` is one of
            ``BltStrategy``, ``BsrStrategy``, ``BisrStrategy``,
            ``LambdaCgdStrategy`` (correlated MF) or ``IdentityStrategy``
            (uncorrelated baseline).
        num_bins: Bins per epoch (k ≥ 2).
        n_steps: Total training rounds.  Must be a positive multiple of
            ``num_bins`` (per-bin participation = ``n_steps // num_bins``).

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        from opaque.dpftrl.noise import blt_strategy, identity_strategy

        # Correlated MF
        s = blt_strategy(n_steps=1000, min_sep=100, max_participations=10)
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=100, n_steps=1000,
        )

        # Identity baseline
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            num_bins=100, n_steps=1000,
        )
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, MfGaussian):
        raise InputTypeError(
            *(
                f"balls_in_bins() requires an MfGaussian inner, got {type(inner).__name__}.",
            )
        )
    if not isinstance(inner.strategy, (*_CorrelatedStrategies, IdentityStrategy)):
        raise InputTypeError(
            *(
                "balls_in_bins() requires inner.strategy in {BltStrategy, "
                "BsrStrategy, BisrStrategy, LambdaCgdStrategy, IdentityStrategy}, "
                f"got {type(inner.strategy).__name__}.",
            )
        )
    # num_bins/n_steps bounds live in ``BallsInBins.__post_init__``.
    return BallsInBins(inner=inner, num_bins=num_bins, n_steps=n_steps)
