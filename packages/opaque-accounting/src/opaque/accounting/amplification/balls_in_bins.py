"""Balls-in-Bins amplification — always returns **total** multi-epoch cost.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins each epoch. Each bin is
processed once, so every example participates exactly once per epoch.

The returned process represents the **total** privacy cost across all
``num_epochs`` epochs.  Do NOT compose further with ``* num_epochs``.

For independent-noise mechanisms (Gaussian, AdaClip), uses a conservative
Poisson per-step approximation.

For correlated-noise mechanisms (DP-λCGD, BISR), uses Monte Carlo sampling of
the dominating pair from Choquette-Choo et al. (2024) arxiv:2410.06266.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian
from opaque.accounting.mechanisms.bisr import Bisr
from opaque.accounting.mechanisms.bsr import Bsr
from opaque.accounting.mechanisms.blt import Blt
from opaque.accounting.mechanisms.lambda_cgd import LambdaCgd
from opaque.accounting.mechanisms.nonprivate import NonPrivate
from opaque.accounting.transformations.adaclip import AdaClip
from opaque.accounting.transformations.second_moment import SecondMoment

#: MF types with pre-computed Gram matrix for MC BnB.
_BnbMf = Blt | LambdaCgd | Bisr | Bsr

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = Gaussian | _BnbMf | AdaClip | SecondMoment | NonPrivate


@dataclass(frozen=True, slots=True)
class BallsInBins(DpProcess):
    """Balls-in-Bins amplified mechanism — **total** multi-epoch cost.

    The returned PLD covers all ``num_epochs`` epochs.
    Do NOT compose further with ``* num_epochs``.

    Example (Gaussian)::

        training = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100, num_epochs=10)
        eps = training.epsilon_at(1e-5)  # total cost, 10 epochs

    Example (DP-λCGD)::

        training = acc.balls_in_bins(
            acc.lambda_cgd(nm, sensitivity=s.sensitivity,
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
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                return _native.balls_in_bins_gaussian_pld_epochs(
                    nm, self.num_bins, self.num_epochs, native_cfg
                )
            case AdaClip(inner=Gaussian()) as ac:
                return _native.balls_in_bins_gaussian_pld_epochs(
                    ac.effective_noise_multiplier,
                    self.num_bins,
                    self.num_epochs,
                    native_cfg,
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
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
            case SecondMoment(inner=Blt() | LambdaCgd() | Bisr() | Bsr()) as second:
                if not second.gram_matrix:
                    raise ValueError(
                        f"SecondMoment({type(second.inner).__name__}) requires a non-empty "
                        "gram_matrix for BnB amplification."
                    )
                return _native.bnb_mc_pld(
                    list(second.gram_matrix),
                    self.num_bins,
                    second.noise_multiplier,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "BallsInBins requires a Gaussian, Blt, LambdaCgd, Bisr, Bsr, "
                    f"AdaClip, or SecondMoment inner mechanism, got {type(self.inner).__name__}."
                )


def balls_in_bins(
    inner: _Inner,
    num_bins: int,
    num_epochs: int = 1,
) -> BallsInBins:
    """Balls-in-Bins amplified mechanism — returns **total** multi-epoch cost.

    Each epoch, the dataset is randomly partitioned into ``num_bins``
    equally-sized bins.  Each bin is processed with the inner mechanism.
    Every example participates exactly once per epoch.

    The returned process covers **all** ``num_epochs`` epochs.
    Do NOT compose further with ``* num_epochs``.

    For independent-noise mechanisms (Gaussian, AdaClip), ``num_epochs``
    controls how many epochs to compose.  For correlated-noise mechanisms
    (DP-λCGD), the epoch count is already encoded in the inner
    mechanism's ``n_steps`` / ``max_participations``, and ``num_epochs``
    serves as validation.

    Args:
        inner: Base mechanism — :func:`gaussian`, :func:`lambda_cgd`,
            :func:`bisr`, :func:`bsr`, or :func:`adaclip`.
        num_bins: Bins per epoch (k ≥ 2).  Typically ``dataset_size / batch_size``.
        num_epochs: Number of training epochs (default 1).

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        training = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100, num_epochs=10)
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(
        inner,
        (Gaussian, Blt, LambdaCgd, Bisr, Bsr, AdaClip, SecondMoment, NonPrivate),
    ):
        raise TypeError(
            f"balls_in_bins() requires a Gaussian, Blt, LambdaCgd, Bisr, Bsr, "
            f"AdaClip, SecondMoment, or NonPrivate inner mechanism, got {type(inner).__name__}. "
            "Example: acc.balls_in_bins(acc.gaussian(nm), num_bins=k, num_epochs=E)"
        )
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2 for BnB amplification, got {num_bins}")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    return BallsInBins(inner=inner, num_bins=num_bins, num_epochs=num_epochs)
