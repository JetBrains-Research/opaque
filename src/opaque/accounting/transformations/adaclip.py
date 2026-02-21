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

Batch size in training vs. calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*  **Training loop**: pass the exact ``clip_state.batch_size`` (which is
   summed across ranks in distributed training).
*  **Calibration** (batch size unknown a priori):

   - Standard Poisson: use ``sample_rate × dataset_size`` (expected m).
   - Truncated Poisson: use ``batch_size_cap`` (pessimistic maximum).

   Always verify the actual privacy spend at runtime.  The accountant
   can check per-step whether the budget is exceeded; the training loop
   should stop early if so.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class AdaClip(DpProcess):
    """Adaptive clipping transformation (Andrew et al. 2021).

    Stores the noise multiplier on the *fraction* (``quantile_noise_multiplier``)
    and the ``batch_size`` needed to resolve the absolute σ_b for privacy
    accounting.  The :attr:`effective_noise_multiplier` property computes the
    adjusted noise multiplier that accounts for the extra privacy cost of
    the quantile estimator.

    When wrapped in a Poisson amplification layer, only the
    ``effective_noise_multiplier`` matters — the wrapper does not need to
    understand AdaClip internals.
    """

    inner: Gaussian
    quantile_noise_multiplier: float
    batch_size: float

    @property
    def effective_noise_multiplier(self) -> float:
        """Noise multiplier adjusted for the quantile estimator’s privacy cost.

        This is the z_eff from Andrew et al. 2021 Theorem 1:

        .. math::

            z_{\\text{eff}} = \\frac{1}{\\sqrt{1/z^2 + 1/(4\\,\\sigma_b^2)}}

        where *z* is the base noise multiplier and
        *σ_b = batch_size × quantile_noise_multiplier*.

        Callers (e.g. Poisson amplification wrappers) should use this value
        instead of the base ``inner.noise_multiplier``.
        """
        sigma_b = self.batch_size * self.quantile_noise_multiplier
        s = _native.adaclip_sensitivity(self.inner.noise_multiplier, sigma_b)
        return 1.0 / s

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque.accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        match self.inner:
            case Gaussian(noise_multiplier=nm):
                z_eff = self.effective_noise_multiplier
                return _native.gaussian_pld(z_eff, config.to_native())
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

    where *z* is the base noise multiplier and *σ_b = batch_size × multiplier*.

    The resulting ``AdaClip`` process exposes
    :attr:`~AdaClip.effective_noise_multiplier` which encapsulates this
    formula.  Poisson amplification wrappers use that property directly;
    they do **not** need to know about AdaClip internals.

    Calibration guidance
    ~~~~~~~~~~~~~~~~~~~~
    During calibration the exact per-step batch size is unknown.  Use
    an approximation and verify the budget at runtime:

    - **Standard Poisson**: ``batch_size = sample_rate × dataset_size``
      (expected batch size).
    - **Truncated Poisson**: ``batch_size = batch_size_cap`` (pessimistic
      maximum).

    The accountant can check whether the budget is exceeded after each
    training step using the *actual* batch size from
    ``clip_state.batch_size``.  The training loop should stop early if
    the budget is exceeded.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        quantile_noise_multiplier: Noise std on the clipping *fraction*
            (value in [0, 1]).  Andrew et al. recommend 1/20 = 0.05,
            corresponding to σ_b = m/20 on the clipped-count sum.
        batch_size: Number of examples per training step.  Used to convert
            the fraction-level multiplier to the absolute σ_b needed by
            ``adaclip_sensitivity()``.  In the training loop pass the
            actual batch size from ``clip_state.batch_size``; during
            calibration use the expected or pessimistic batch size.

    Returns:
        An :class:`AdaClip` process with an
        :attr:`~AdaClip.effective_noise_multiplier` property.

    Example::

        # --- In training loop (exact batch size from state) ---
        step = acc.adaclip(
            acc.gaussian(noise_multiplier),
            batch_size=clip_state.batch_size,   # actual
        )
        accountant = accountant | step

        # --- During calibration (approximate batch size) ---
        batch_approx = sample_rate * dataset_size          # Poisson
        # or: batch_approx = batch_size_cap                # Truncated Poisson
        step = acc.poisson(
            acc.adaclip(acc.gaussian(1.1), batch_size=batch_approx),
            sample_rate=0.01,
        )
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
