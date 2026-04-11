"""Balls-in-Bins amplification for Gaussian mechanism.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins each epoch. Each bin is
processed once with a Gaussian mechanism, so every example participates
exactly once per epoch.

This provides privacy amplification because the adversary does not know
which bin contains the target example. The per-epoch PLD is computed using
a conservative Poisson per-step approximation composed ``num_bins`` times.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Privacy Amplification for Matrix Mechanisms"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.lambda_cgd import LambdaCgd
from opaque_accounting.mechanisms.nonprivate import NonPrivate
from opaque_accounting.transformations.adaclip import AdaClip

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = Gaussian | LambdaCgd | AdaClip | NonPrivate


@dataclass(frozen=True, slots=True)
class BallsInBins(DpProcess):
    """Balls-in-Bins amplified Gaussian mechanism.

    The dataset is partitioned into ``num_bins`` bins each epoch.
    Each bin is processed with the inner Gaussian mechanism.

    For mechanisms without cross-epoch correlations (Gaussian, AdaClip),
    the PLD represents one epoch — multiply by the number of epochs::

        epoch = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100)
        training = epoch * 10  # 10 epochs

    For λCGD, pass the full multi-epoch parameters.  The PLD is computed
    via Monte Carlo sampling of the BnB dominating pair (arxiv:2410.06266).
    This IS the total privacy cost — do NOT compose with ``* num_epochs``::

        training = acc.balls_in_bins(
            acc.lambda_cgd(nm, lambda_=0.9, n_steps=total_steps,
                           min_sep=steps_per_epoch,
                           max_participations=num_epochs),
            num_bins=steps_per_epoch,
        )
        eps = training.epsilon_at(1e-5)  # total cost
    """

    inner: _Inner
    num_bins: int

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque_accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        native_cfg = config.to_native()

        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                return _native.balls_in_bins_gaussian_pld(
                    nm, self.num_bins, native_cfg
                )
            case AdaClip(inner=Gaussian()) as ac:
                return _native.balls_in_bins_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.num_bins,
                    native_cfg,
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case LambdaCgd() as lc:
                # Monte Carlo BnB accounting (Lemma 3.2 of arxiv:2410.06266).
                # Compute Gram matrix of the dominating pair mixture means
                # from the full multi-epoch C_λ parameters, then sample
                # the PLD via Monte Carlo.  This is the paper-correct
                # approach — no per-epoch composition needed.
                gram = _native.lambda_cgd_gram_matrix(
                    lc.lambda_,
                    lc.n_steps,
                    lc.min_sep,
                    lc.max_participations,
                    lc.normalized,
                )
                return _native.bnb_mc_pld(
                    gram,
                    self.num_bins,
                    lc.noise_multiplier,
                    100_000,  # MC samples (100K: fast + accurate)
                    42,  # seed
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "BallsInBins requires a Gaussian, LambdaCgd, or AdaClip inner "
                    f"mechanism, got {type(self.inner).__name__}."
                )


def balls_in_bins(
    inner: _Inner,
    num_bins: int,
) -> BallsInBins:
    """Balls-in-Bins amplified Gaussian mechanism.

    Each epoch, the dataset is randomly partitioned into ``num_bins``
    equally-sized bins. Each bin is processed with the inner Gaussian
    mechanism. Every example participates exactly once per epoch.

    For simple mechanisms (Gaussian, AdaClip), the returned process
    represents one epoch.  Multiply by the number of epochs::

        epoch = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100)
        training = epoch * 10  # 10 epochs

    For :func:`lambda_cgd` with reused bin allocation, encode the
    full multi-epoch participation pattern and do NOT compose::

        training = acc.balls_in_bins(
            acc.lambda_cgd(nm, lambda_=0.9, n_steps=total_steps,
                           min_sep=steps_per_epoch,
                           max_participations=num_epochs),
            num_bins=steps_per_epoch,
        )

    Args:
        inner: The base mechanism — :func:`gaussian` or :func:`adaclip`.
        num_bins: Number of bins (k ≥ 2). Typically ``dataset_size / batch_size``.

    Returns:
        A :class:`BallsInBins` process.

    Example::

        epoch = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100)
        eps = (epoch * 10).epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, LambdaCgd, AdaClip, NonPrivate)):
        raise TypeError(
            f"balls_in_bins() requires a Gaussian, LambdaCgd, AdaClip, or NonPrivate "
            f"inner mechanism, got {type(inner).__name__}. "
            "Example: acc.balls_in_bins(acc.gaussian(nm), num_bins=k)"
        )
    if num_bins < 2:
        raise ValueError(
            f"num_bins must be >= 2 for BnB amplification, got {num_bins}"
        )
    return BallsInBins(inner=inner, num_bins=num_bins)
