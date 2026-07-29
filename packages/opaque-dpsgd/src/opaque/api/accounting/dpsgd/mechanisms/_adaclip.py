"""Adaptive clipping transformation for privacy accounting.

Accounts for the extra privacy cost of the noised clipping-fraction
query used by adaptive gradient clipping.  The ``fraction_noise_std``
parameter controls the noise on the fraction (default 0.05); the
absolute noise std is ``σ_b = expected_batch_size × fraction_noise_std``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian

#: Mechanism types accepted as AdaClip inner.
_Inner = Gaussian | NonPrivate


@dataclass(frozen=True, slots=True)
class AdaClip(DpProcess):
    """Adaptive clipping transformation.

    Wraps an ``inner`` mechanism and adds the privacy cost of the noised
    clipping-fraction query.  ``σ_b = expected_batch_size × fraction_noise_std``.

    When ``num_groups > 1``, accounts for ``K`` independent quantile queries
    (one per parameter group in per-group adaptive clipping).
    """

    inner: _Inner
    fraction_noise_std: float
    expected_batch_size: float
    num_groups: int = 1

    @property
    def effective_noise_multiplier(self) -> float:
        """Noise multiplier adjusted for the quantile estimator's privacy cost.

        Exact for Gaussian ``inner``; conservative for bounded Gaussian.
        Returns ``0.0`` for :class:`NonPrivate` inner (no noise).

        When ``num_groups > 1``, the effective noise multiplier is lower
        (more privacy consumed) due to ``K`` independent quantile queries.
        """
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return 0.0
            case Gaussian(noise_multiplier=nm):
                sigma_b = self.expected_batch_size * self.fraction_noise_std
                s = _native.adaclip_sensitivity(nm, sigma_b, self.num_groups)
                return 1.0 / s

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

        native_cfg = config.to_native()

        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian():
                return _native.gaussian_pld(self.effective_noise_multiplier, native_cfg)
            case _:
                inner_pld = self.inner.pld(
                    discretization=discretization,
                    log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                    max_grid_size=max_grid_size,
                )
                sigma_b = self.expected_batch_size * self.fraction_noise_std
                bit_pld = _native.gaussian_pld(2.0 * sigma_b, native_cfg)
                if self.num_groups > 1:
                    bit_pld = bit_pld * self.num_groups
                return inner_pld.compose(bit_pld)


def adaclip(
    inner: _Inner,
    *,
    fraction_noise_std: float = 0.05,
    expected_batch_size: float,
    num_groups: int = 1,
) -> AdaClip:
    """Account for the privacy cost of adaptive clipping.

    Wraps an ``inner`` mechanism and adds the cost of the noised
    clipping-fraction query.

    Args:
        inner: Base mechanism — ``gaussian()``.
        fraction_noise_std: Noise std on the clipping fraction
            (default 0.05).
        expected_batch_size: Data-independent batch size
            (``sample_rate × dataset_size`` under Poisson sampling).
        num_groups: Number of independent quantile queries (default 1).
            Set to ``K`` for per-group adaptive clipping with ``K`` groups.

    Returns:
        An :class:`AdaClip` process with an
        :attr:`~AdaClip.effective_noise_multiplier` property.

    Example::

        expected_bs = sample_rate * dataset_size

        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.1), expected_batch_size=expected_bs),
            sample_rate=0.01,
        )
    """
    match inner:
        case Gaussian() | NonPrivate():
            pass
        case _:
            raise TypeError(
                f"adaclip() requires a Gaussian or NonPrivate inner mechanism, "
                f"got {type(inner).__name__}."
            )
    if fraction_noise_std <= 0:
        raise ValueError(
            f"fraction_noise_std must be positive, got {fraction_noise_std}"
        )
    if expected_batch_size <= 0:
        raise ValueError(
            f"expected_batch_size must be positive, got {expected_batch_size}"
        )
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")

    return AdaClip(
        inner=inner,
        fraction_noise_std=fraction_noise_std,
        expected_batch_size=expected_batch_size,
        num_groups=num_groups,
    )
