"""Balls-in-Bins amplification for DP-FTRL — **total** multi-epoch cost.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins.  The bin assignment
is fixed once at sampler init and reused across all ``num_epochs``
epochs, so each example stays in its bin — required for the
dominating-pair analysis.

The Choquette-Choo et al. (2024) dominating pair (Lemma 3.2) is::

    P = (1/b) Σ_{i=1}^{b} N(m_i, σ²I)        m_i = Σ_{j=0}^{E-1} |C|[:, b·j + i]
    Q = N(0, σ²I)

After Gram-matrix reduction (``G[i,j] = m_i · m_j``) the privacy loss only
depends on ``G``, ``num_bins`` and ``σ``.  Both dispatch paths feed this
construction:

- **Correlated-noise** (matrix-factorisation): ``Blt``, ``LambdaCgd``,
  ``Bisr``, ``Bsr`` — pass the strategy's pre-computed Gram matrix.
- **MF identity** (uncorrelated noise — :class:`IdentityMf`): ``C = I``
  gives orthogonal ``m_i`` with ``‖m_i‖² = num_epochs``, i.e.
  ``G = num_epochs · I_b`` (diagonal).  This feeds the same Lemma 3.2
  dominating pair through Monte Carlo — a valid bound on the BnB
  mechanism's privacy.  In standard regimes the heuristic per-step
  Poisson approximation
  ``poisson(gaussian(σ), 1/num_bins) * (num_bins * num_epochs)`` happens
  to be numerically tighter, but it is not a strict upper bound on
  shuffle-style BnB sampling (Chua et al. 2024).  Lemma 3.2 is the
  rigorous bound and is used here.

The returned process represents the **total** privacy cost across
all ``num_epochs`` epochs.  Do NOT compose further with ``* num_epochs``.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native
from opaque.accounting._base import DpProcess, Pld

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = DpProcess


@dataclass(frozen=True, slots=True)
class BallsInBins(DpProcess):
    """Balls-in-Bins amplified MF mechanism — **total** multi-epoch cost.

    The returned PLD covers all ``num_epochs`` epochs.
    Do NOT compose further with ``* num_epochs``.

    Example (DP-λCGD)::

        training = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(nm, sensitivity=s.sensitivity,
                                gram_matrix=s.gram_matrix),
            num_bins=steps_per_epoch,
            num_epochs=num_epochs,
        )
        eps = training.epsilon_at(1e-5)  # total cost
    """

    inner: _Inner
    num_bins: int
    num_epochs: int

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
        from opaque.dpftrl.accounting.mechanisms._bisr import Bisr
        from opaque.dpftrl.accounting.mechanisms._blt import Blt
        from opaque.dpftrl.accounting.mechanisms._bsr import Bsr
        from opaque.dpftrl.accounting.mechanisms._identity import IdentityMf
        from opaque.dpftrl.accounting.mechanisms._lambda_cgd import LambdaCgd
        from opaque.accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )

        native_cfg = config.to_native()

        match self.inner:
            case Blt() | LambdaCgd() | Bisr() | Bsr() as mg:
                if not mg.gram_matrix:
                    raise ValueError(
                        f"{type(mg).__name__} requires a non-empty gram_matrix "
                        "for BnB amplification."
                    )
                return _native.bnb_mc_pld(
                    list(mg.gram_matrix),
                    self.num_bins,
                    mg.noise_multiplier,
                    native_cfg,
                )
            case IdentityMf() as mf_id:
                # Identity (C = I) ⇒ Lemma 3.2 m_i are orthogonal with
                # ‖m_i‖² = num_epochs ⇒ Gram = num_epochs · I_b.
                if mf_id.noise_multiplier == 0:
                    return _native.non_private_pld(native_cfg)
                b = self.num_bins
                gram_diag = float(self.num_epochs)
                gram_flat = [
                    gram_diag if i == j else 0.0
                    for i in range(b)
                    for j in range(b)
                ]
                return _native.bnb_mc_pld(
                    gram_flat,
                    b,
                    float(mf_id.noise_multiplier),
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "BallsInBins requires Blt, LambdaCgd, Bisr, Bsr, or "
                    "IdentityMf inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def balls_in_bins(
    inner: _Inner,
    num_bins: int,
    num_epochs: int = 1,
) -> BallsInBins:
    """Balls-in-Bins amplified MF mechanism — **total** multi-epoch cost.

    Each epoch, the dataset is partitioned into ``num_bins`` bins (assignment
    fixed at sampler init and reused across epochs).  Every example
    participates exactly once per epoch.  The returned process covers all
    ``num_epochs`` epochs — do NOT compose further with ``* num_epochs``.

    Accepted inner mechanisms:

    - **Correlated-noise (matrix-factorisation)**: :func:`blt`, :func:`lambda_cgd`,
      :func:`bisr`, :func:`bsr` — PLD via the Monte Carlo dominating-pair
      analysis (Choquette-Choo et al. 2024).
    - **MF identity** (:func:`mf_identity`) — tight process-level reduction
      using the bin-aggregation equivalence (E rounds per bin collapse to one
      Gaussian at ``σ / √E``) plus Poisson at the bin granularity.

    Args:
        inner: An MF mechanism — :func:`blt`, :func:`lambda_cgd`, :func:`bisr`,
            :func:`bsr`, or :func:`mf_identity`.
        num_bins: Bins per epoch (k ≥ 2).  Typically ``dataset_size / batch_size``.
        num_epochs: Number of training epochs (default 1).  For correlated-
            noise mechanisms the epoch count is already encoded in the inner
            mechanism's ``n_steps`` / ``max_participations`` — ``num_epochs``
            serves as validation.  For ``IdentityMf`` it is the actual epoch
            count used in the tight reduction.

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        # Correlated MF
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(nm, sensitivity=s.sensitivity,
                                gram_matrix=s.gram_matrix),
            num_bins=100,
            num_epochs=10,
        )

        # Identity baseline through the FTRL training loop
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(1.0), num_bins=100, num_epochs=10,
        )
        eps = training.epsilon_at(1e-5)
    """
    from opaque.dpftrl.accounting.mechanisms._bisr import Bisr
    from opaque.dpftrl.accounting.mechanisms._blt import Blt
    from opaque.dpftrl.accounting.mechanisms._bsr import Bsr
    from opaque.dpftrl.accounting.mechanisms._identity import IdentityMf
    from opaque.dpftrl.accounting.mechanisms._lambda_cgd import LambdaCgd

    match inner:
        case Blt() | LambdaCgd() | Bisr() | Bsr() | IdentityMf():
            pass
        case _:
            raise TypeError(
                "balls_in_bins() requires Blt, LambdaCgd, Bisr, Bsr, or "
                f"IdentityMf inner mechanism, got {type(inner).__name__}."
            )
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2 for BnB amplification, got {num_bins}")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    return BallsInBins(inner=inner, num_bins=num_bins, num_epochs=num_epochs)
