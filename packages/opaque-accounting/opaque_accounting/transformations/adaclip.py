"""Adaptive clipping transformation (Andrew et al. 2021).

Noise parameterisation
~~~~~~~~~~~~~~~~~~~~~~
Andrew et al. add Gaussian noise to the *count* of clipped examples:

    b̃_t = (1/m)(Σ b_i + N(0, σ_b²))

where m is the batch size and σ_b = m/20 is the paper's default.
Dividing through, noise on the *fraction* is σ_b / m = 1/20 = 0.05,
independent of batch size.

We store the fraction-level multiplier ``quantile_noise_multiplier``
(default 0.05, matching Andrew et al.).  The conversion to the absolute
``σ_b`` that ``adaclip_sensitivity()`` needs requires a batch size:

    σ_b = batch_size × quantile_noise_multiplier

The ``AdaClip`` process exposes :attr:`effective_noise_multiplier` which
encapsulates the combined sensitivity formula.  Poisson amplification
wrappers use that property directly — they do not need to understand
AdaClip internals.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config
from opaque_accounting.mechanisms.gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class AdaClip(DpProcess):
    """Adaptive clipping transformation (Andrew et al. 2021).

    Stores the noise multiplier on the *fraction* (``quantile_noise_multiplier``)
    and the ``batch_size`` needed to resolve the absolute σ_b for privacy
    accounting.  The :attr:`effective_noise_multiplier` property computes the
    adjusted noise multiplier that accounts for the extra privacy cost of
    the quantile estimator.
    """

    inner: Gaussian
    quantile_noise_multiplier: float
    batch_size: float

    @property
    def effective_noise_multiplier(self) -> float:
        """Noise multiplier adjusted for the quantile estimator's privacy cost.

        .. math::

            z_{\\text{eff}} = \\frac{1}{\\sqrt{1/z^2 + 1/(4\\,\\sigma_b^2)}}

        where *z* is the base noise multiplier and
        *σ_b = batch_size × quantile_noise_multiplier*.
        """
        sigma_b = self.batch_size * self.quantile_noise_multiplier
        s = _native.adaclip_sensitivity(self.inner.noise_multiplier, sigma_b)
        return 1.0 / s

    def pmf(self, **kwargs: object) -> PmfPld:
        match self.inner:
            case Gaussian():
                z_eff = self.effective_noise_multiplier
                return PmfPld(_native.gaussian_pld(z_eff, _make_native_config(**kwargs)))
            case _:
                raise TypeError(
                    "AdaClip requires a Gaussian inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def adaclip(
    inner: Gaussian,
    *,
    quantile_noise_multiplier: float = 0.05,
    batch_size: float,
) -> AdaClip:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the observed
    fraction of clipped gradients.  The fraction estimate is made private by
    adding Gaussian noise, which costs a small additional privacy budget.

    The total privacy cost uses the combined sensitivity formula (Theorem 1):

    .. math::

        z_{\\text{eff}} = \\frac{1}{\\sqrt{1/z^2 + 1/(4\\,\\sigma_b^2)}}

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        quantile_noise_multiplier: Noise std on the clipping *fraction*.
            Default 0.05 (matching Andrew et al.).
        batch_size: Number of examples per training step.

    Returns:
        An :class:`AdaClip` process.

    Example::

        step = acc.poisson(
            acc.adaclip(acc.gaussian(1.1), batch_size=500),
            sample_rate=0.01,
        )
        training = step * 1000
        eps = training.cgf().epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"adaclip() requires a Gaussian inner mechanism, got {type(inner).__name__}."
        )
    if quantile_noise_multiplier <= 0:
        raise ValueError(
            "quantile_noise_multiplier must be positive, "
            f"got {quantile_noise_multiplier}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return AdaClip(
        inner=inner,
        quantile_noise_multiplier=quantile_noise_multiplier,
        batch_size=batch_size,
    )
