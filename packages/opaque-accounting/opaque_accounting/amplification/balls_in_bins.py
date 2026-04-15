"""Balls-in-Bins amplification — always returns **total** multi-epoch cost.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins each epoch. Each bin is
processed once, so every example participates exactly once per epoch.

The returned process represents the **total** privacy cost across all
``num_epochs`` epochs.  Do NOT compose further with ``* num_epochs``.

For independent-noise mechanisms (Gaussian, AdaClip), uses a conservative
Poisson per-step approximation.

For correlated-noise mechanisms (DP-λCGD, BISR, BLT), uses either Monte Carlo
sampling of the dominating pair from Choquette-Choo et al. (2024)
arxiv:2410.06266 (default), or a deterministic moment-based envelope
(Schuchardt & Kalinin, 2026 arxiv:2601.21636) when ``method="deterministic"``.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
    - Schuchardt & Kalinin (2026), "Sampling-Free Privacy Accounting for
      Matrix Mechanisms under Random Allocation"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.bisr import Bisr
from opaque_accounting.mechanisms.blt import Blt
from opaque_accounting.mechanisms.lambda_cgd import LambdaCgd
from opaque_accounting.mechanisms.nonprivate import NonPrivate
from opaque_accounting.transformations.adaclip import AdaClip
from opaque_accounting.transformations.jme import Jme

#: MF types with pre-computed Gram matrix for BnB accounting.
_BnbMf = Blt | LambdaCgd | Bisr

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = Gaussian | _BnbMf | AdaClip | Jme | NonPrivate

BnbMethod = Literal["mc", "deterministic"]


@dataclass(frozen=True, slots=True)
class DeterministicOptions:
    """Parameters for sampling-free BnB accounting (matrix mechanisms only)."""

    max_order_k: int = 12
    epsilon_max: float = 20.0
    epsilon_points: int = 256
    max_states: int = 200_000


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
    method: BnbMethod = "mc"
    deterministic_options: DeterministicOptions | None = None

    def _bnb_native_kwargs(self) -> dict[str, Any]:
        if self.deterministic_options is None:
            return {}
        d = self.deterministic_options
        return {
            "max_order_k": d.max_order_k,
            "epsilon_max": d.epsilon_max,
            "epsilon_points": d.epsilon_points,
            "max_states": d.max_states,
        }

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
            case Blt() | LambdaCgd() | Bisr() as mg:
                if not mg.gram_matrix:
                    raise ValueError(
                        f"{type(mg).__name__} requires a non-empty gram_matrix "
                        "for BnB amplification."
                    )
                if self.method == "mc":
                    return _native.bnb_mc_pld(
                        list(mg.gram_matrix),
                        self.num_bins,
                        mg.noise_multiplier,
                        100_000,  # MC samples
                        42,  # seed
                        native_cfg,
                    )
                return _native.bnb_deterministic_pld(
                    list(mg.gram_matrix),
                    self.num_bins,
                    mg.noise_multiplier,
                    native_cfg,
                    **self._bnb_native_kwargs(),
                )
            case Jme(inner=Blt() | LambdaCgd() | Bisr()) as j:
                if not j.gram_matrix:
                    raise ValueError(
                        f"Jme({type(j.inner).__name__}) requires a non-empty "
                        "gram_matrix for BnB amplification."
                    )
                if self.method == "mc":
                    return _native.bnb_mc_pld(
                        list(j.gram_matrix),
                        self.num_bins,
                        j.noise_multiplier,
                        100_000,  # MC samples
                        42,  # seed
                        native_cfg,
                    )
                return _native.bnb_deterministic_pld(
                    list(j.gram_matrix),
                    self.num_bins,
                    j.noise_multiplier,
                    native_cfg,
                    **self._bnb_native_kwargs(),
                )
            case _:
                raise TypeError(
                    "BallsInBins requires a Gaussian, Blt, LambdaCgd, Bisr, "
                    f"AdaClip, or Jme inner mechanism, got {type(self.inner).__name__}."
                )


def balls_in_bins(
    inner: _Inner,
    num_bins: int,
    num_epochs: int = 1,
    *,
    method: BnbMethod = "mc",
    deterministic_options: DeterministicOptions | None = None,
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
            :func:`bisr`, or :func:`adaclip`.
        num_bins: Bins per epoch (k >= 2).  Typically ``dataset_size / batch_size``.
        num_epochs: Number of training epochs (default 1).
        method: ``"mc"`` (default) or ``"deterministic"`` for matrix mechanisms.
        deterministic_options: Optional tuning for deterministic accounting.

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        training = acc.balls_in_bins(acc.gaussian(1.1), num_bins=100, num_epochs=10)
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(
        inner,
        (Gaussian, Blt, LambdaCgd, Bisr, AdaClip, Jme, NonPrivate),
    ):
        raise TypeError(
            f"balls_in_bins() requires a Gaussian, Blt, LambdaCgd, Bisr, "
            f"AdaClip, Jme, or NonPrivate inner mechanism, got {type(inner).__name__}. "
            "Example: acc.balls_in_bins(acc.gaussian(nm), num_bins=k, num_epochs=E)"
        )
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2 for BnB amplification, got {num_bins}")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    if method not in ("mc", "deterministic"):
        raise ValueError(f"method must be 'mc' or 'deterministic', got {method!r}")

    return BallsInBins(
        inner=inner,
        num_bins=num_bins,
        num_epochs=num_epochs,
        method=method,
        deterministic_options=deterministic_options,
    )
