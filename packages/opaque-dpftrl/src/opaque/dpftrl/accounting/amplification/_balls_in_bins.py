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

- **Correlated-noise** (matrix-factorisation): ``Blt``, ``LambdaCgd``,
  ``Bisr``, ``Bsr`` — pass the strategy's pre-computed Gram matrix.
- **MF identity** (uncorrelated noise — :class:`IdentityMf`): ``C = I``
  gives orthogonal ``m_i`` with ``‖m_i‖² = E``, i.e. ``G = E · I_b``
  (diagonal).  This feeds the same Lemma 3.2 dominating pair through
  Monte Carlo — a rigorous bound on the BnB mechanism's privacy.

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

from opaque.accounting import _native
from opaque.accounting._base import DpProcess, Pld

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = DpProcess


#: Importance-sampling tilt used by :func:`bnb_mc_pld_identity` for the
#: ``IdentityMf`` dispatch.  Hardcoded to ``1.0``: empirically robust across
#: DP-FTRL training regimes (ε ∈ [0.5, 20], σ ∈ [0.5, 3], k ∈ [8, 1000],
#: E ∈ [1, 16]) — gives 4-76× MC variance reduction vs no IS in 9/10 swept
#: configs and only ~3× worse (still ≤ 2.5% rel σ at 500k samples, so still
#: tight in absolute terms) in the heavy-noise / very-low-ε edge case.
#: Fixing this in code rather than exposing as a knob: the value is an MC
#: internal detail, not a mechanism property; treating it like a privacy
#: parameter would mislead users.  If a degenerate config ever surfaces,
#: this is the place to revisit (or add a ``pld()`` kwarg as an escape).
_IDENTITY_IS_TILT: float = 1.0


@dataclass(frozen=True, slots=True)
class BallsInBins(DpProcess):
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    The returned PLD covers all ``n_steps`` training rounds (= ``num_bins``
    bins × ``n_steps // num_bins`` epochs).  Do NOT compose externally.

    For ``IdentityMf`` inner, the dispatch uses
    :func:`opaque.accounting._native.bnb_mc_pld_identity` — a specialised
    importance-sampled MC that exploits the diagonal Gram structure
    (`G = num_epochs · I_b`).  The IS tilt is fixed internally
    (``_IDENTITY_IS_TILT``); see that constant's docstring.

    Example (DP-λCGD)::

        training = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(nm, sensitivity=s.sensitivity,
                                gram_matrix=s.gram_matrix),
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
                # Specialised primitive skips Cholesky, fixes shifted bin
                # to index 0 by symmetry, and applies importance sampling
                # on the shifted-bin coordinate.
                if mf_id.noise_multiplier == 0:
                    return _native.non_private_pld(native_cfg)
                return _native.bnb_mc_pld_identity(
                    self.num_bins,
                    self.num_epochs,
                    float(mf_id.noise_multiplier),
                    _IDENTITY_IS_TILT,
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

    Accepted inner mechanisms:

    - **Correlated-noise (matrix-factorisation)**: :func:`blt`, :func:`lambda_cgd`,
      :func:`bisr`, :func:`bsr` — PLD via the Monte Carlo dominating-pair
      analysis (Choquette-Choo et al. 2024).
    - **MF identity** (:func:`mf_identity`) — same Lemma 3.2 dominating pair
      with ``Gram = num_epochs · I_b`` (orthogonal supports), via the
      specialised importance-sampled MC primitive.  The IS tilt is fixed
      internally to a value that's empirically robust across DP-FTRL
      training regimes; see ``_IDENTITY_IS_TILT`` for the rationale.

    Args:
        inner: An MF mechanism — :func:`blt`, :func:`lambda_cgd`, :func:`bisr`,
            :func:`bsr`, or :func:`mf_identity`.
        num_bins: Bins per epoch (k ≥ 2).
        n_steps: Total training rounds.  Must be a positive multiple of
            ``num_bins`` (per-bin participation = ``n_steps // num_bins``).

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        # Correlated MF
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(nm, sensitivity=s.sensitivity,
                                gram_matrix=s.gram_matrix),
            num_bins=100, n_steps=1000,
        )

        # Identity baseline through the FTRL training loop
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(1.0), num_bins=100, n_steps=1000,
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
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_steps % num_bins != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a positive multiple of "
            f"num_bins ({num_bins}); BnB analysis assumes integer epochs."
        )

    return BallsInBins(inner=inner, num_bins=num_bins, n_steps=n_steps)
