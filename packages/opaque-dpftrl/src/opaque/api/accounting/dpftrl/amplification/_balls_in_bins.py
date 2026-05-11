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
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.dpftrl._base import DpFtrlProcess
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._bisr import BisrStrategy
from opaque.api.dpftrl.noise._blt import BltStrategy
from opaque.api.dpftrl.noise._bsr import BsrStrategy
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = MfGaussian

#: Strategy types whose Gram is needed at PLD time (the "correlated MF" set).
_CorrelatedStrategies = (BltStrategy, BsrStrategy, BisrStrategy, LambdaCgdStrategy)


#: Importance-sampling tilt used by :func:`bnb_mc_pld_identity` for the
#: ``IdentityStrategy`` dispatch.  Hardcoded to ``1.0``: empirically robust
#: across DP-FTRL training regimes (ε ∈ [0.5, 20], σ ∈ [0.5, 3], k ∈ [8, 1000],
#: E ∈ [1, 16]) — gives 4-76× MC variance reduction vs no IS in 9/10 swept
#: configs and only ~3× worse (still ≤ 2.5% rel σ at 500k samples) in the
#: heavy-noise / very-low-ε edge case.  Fixing this in code rather than
#: exposing as a knob: the value is an MC internal detail, not a mechanism
#: property; treating it like a privacy parameter would mislead users.
_IDENTITY_IS_TILT: float = 1.0


@dataclass(frozen=True, slots=True)
class BallsInBins(DpFtrlProcess):
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    The returned PLD covers all ``n_steps`` training rounds (= ``num_bins``
    bins × ``n_steps // num_bins`` epochs).  Do NOT compose externally.

    For ``IdentityStrategy`` inner, the dispatch uses
    :func:`opaque.accounting._native.bnb_mc_pld_identity` — a specialised
    importance-sampled MC that exploits the diagonal Gram structure
    (``G = num_epochs · I_b``).

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
        # analysis is defined at epoch boundaries.  ``approx_at_step`` rounds up to
        # the next epoch.
        return self.num_bins

    def approx_at_step(self, step: int) -> DpProcess:
        """Process truncated to its first ``step`` rounds (rounded up to an epoch).

        Correlated-MF strategies regenerate their Gram and sensitivity for
        the shorter horizon via ``strategy.with_horizon``.  ``IdentityStrategy``
        has no Gram and its ``with_horizon`` returns self; the wrapping BnB
        just clamps ``n_steps``.

        See :meth:`DpFtrlProcess.approx_at_step` for the upper-bound
        semantics.
        """
        import dataclasses

        from opaque.api.accounting.core.mechanisms.types import Identity

        if step <= 0:
            return Identity()
        if step >= self.n_steps:
            return self
        unit = self.atomic_unit
        if unit < 1:
            raise ValueError(
                f"{type(self).__name__}.atomic_unit must be >= 1, got {unit}"
            )
        rounded = min(-(-step // unit) * unit, self.n_steps)
        if rounded == self.n_steps:
            return self
        if isinstance(self.inner.strategy, _CorrelatedStrategies):
            new_strategy = self.inner.strategy.with_horizon(
                n_steps=rounded,
                max_participations=rounded // self.num_bins,
            )
            new_inner = dataclasses.replace(self.inner, strategy=new_strategy)
            return dataclasses.replace(self, inner=new_inner, n_steps=rounded)
        return dataclasses.replace(self, n_steps=rounded)

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )
        native_cfg = config.to_native()

        match self.inner.strategy:
            case (
                BltStrategy()
                | LambdaCgdStrategy()
                | BisrStrategy()
                | BsrStrategy() as s
            ):
                if not s._gram_matrix:
                    raise ValueError(
                        f"{type(s).__name__} requires a non-empty gram_matrix "
                        "for BnB amplification."
                    )
                return _native.bnb_mc_pld(
                    list(s._gram_matrix),
                    self.num_bins,
                    self.inner.noise_multiplier,
                    native_cfg,
                )
            case IdentityStrategy():
                # Identity (C = I) ⇒ Lemma 3.2 m_i are orthogonal with
                # ‖m_i‖² = num_epochs ⇒ Gram = num_epochs · I_b.
                # Specialised primitive skips Cholesky, fixes shifted bin
                # to index 0 by symmetry, and applies importance sampling
                # on the shifted-bin coordinate.
                if self.inner.noise_multiplier == 0:
                    return _native.non_private_pld(native_cfg)
                return _native.bnb_mc_pld_identity(
                    self.num_bins,
                    self.num_epochs,
                    float(self.inner.noise_multiplier),
                    _IDENTITY_IS_TILT,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "BallsInBins requires inner.strategy in {BltStrategy, "
                    "BsrStrategy, BisrStrategy, LambdaCgdStrategy, "
                    "IdentityStrategy}, got "
                    f"{type(self.inner.strategy).__name__}."
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
        raise TypeError(
            f"balls_in_bins() requires an MfGaussian inner, got {type(inner).__name__}."
        )
    if not isinstance(inner.strategy, _CorrelatedStrategies + (IdentityStrategy,)):
        raise TypeError(
            "balls_in_bins() requires inner.strategy in {BltStrategy, "
            "BsrStrategy, BisrStrategy, LambdaCgdStrategy, IdentityStrategy}, "
            f"got {type(inner.strategy).__name__}."
        )
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2 for BnB amplification, got {num_bins}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_steps % num_bins != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a positive multiple of "
            f"num_bins ({num_bins}); BnB analysis assumes integer epochs."
        )

    return BallsInBins(inner=inner, num_bins=num_bins, n_steps=n_steps)
