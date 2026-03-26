"""Adaptive clipping transformation (Andrew et al. 2021).

Noise parameterisation
~~~~~~~~~~~~~~~~~~~~~~
Andrew et al. add Gaussian noise to the *count* of clipped examples:

    b̃_t = (1/m)(Σ b_i + N(0, σ_b²))

where m is the batch size and σ_b = m/20 is the paper's default.
Dividing through, noise on the *fraction* is σ_b / m = 1/20 = 0.05,
independent of batch size.

We store the fraction-level std ``fraction_noise_std``
(default 0.05, matching Andrew et al.).  The conversion to the absolute
``σ_b`` that ``adaclip_sensitivity()`` needs requires a batch size:

    σ_b = batch_size × fraction_noise_std

PLD composition
~~~~~~~~~~~~~~~
The adaptive clipping step releases two independent outputs:

1. Clipped-and-noised gradients (via the ``inner`` mechanism).
2. Noised fraction of clipped examples (Gaussian on the centered bit).

The centered bit b_i - 1/2 has L2 sensitivity 1/2 under add/remove
neighbouring.  With noise std σ_b on the count sum, the bit
mechanism's noise multiplier is σ_b / (1/2) = 2 σ_b.

For a Gaussian inner mechanism, the joint PLD reduces to the
closed-form z_eff from Theorem 1 (tight).  For rectified / truncated
Gaussian the two PLDs are composed independently (valid but
conservative).

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

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.rectified_gaussian import RectifiedGaussian
from opaque_accounting.mechanisms.truncated_gaussian import TruncatedGaussian

#: Mechanism types accepted as AdaClip inner.
_Inner = Gaussian | RectifiedGaussian | TruncatedGaussian


@dataclass(frozen=True, slots=True)
class AdaClip(DpProcess):
    """Adaptive clipping transformation (Andrew et al. 2021).

    Stores the noise std on the *fraction* (``fraction_noise_std``)
    and the ``batch_size`` needed to resolve the absolute σ_b for privacy
    accounting.

    For a Gaussian ``inner``, :attr:`effective_noise_multiplier` gives the
    tight z_eff from Theorem 1.  For rectified / truncated Gaussian the
    ``pld()`` method composes the inner mechanism's PLD with the bit
    mechanism's PLD (Gaussian, noise multiplier = 2 σ_b).
    """

    inner: _Inner
    fraction_noise_std: float
    batch_size: float

    @property
    def effective_noise_multiplier(self) -> float:
        """Noise multiplier adjusted for the quantile estimator's privacy cost.

        This is the z_eff from Andrew et al. 2021 Theorem 1:

        .. math::

            z_{\\text{eff}} = \\frac{1}{\\sqrt{1/z^2 + 1/(4\\,\\sigma_b^2)}}

        where *z* is the base noise multiplier and
        *σ_b = batch_size × fraction_noise_std*.

        Only exact when ``inner`` is Gaussian.  Callers (e.g. Poisson
        amplification wrappers) should use this value instead of the base
        ``inner.noise_multiplier``.
        """
        sigma_b = self.batch_size * self.fraction_noise_std
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
        from opaque_accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        native_cfg = config.to_native()

        match self.inner:
            case Gaussian():
                # Tight: z_eff folds both into one Gaussian.
                return _native.gaussian_pld(
                    self.effective_noise_multiplier, native_cfg
                )
            case _:
                # Non-Gaussian: compose inner PLD with bit PLD.
                inner_pld = self.inner.pld(
                    discretization=discretization,
                    log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                    pessimistic_estimate=pessimistic_estimate,
                    max_grid_size=max_grid_size,
                )
                sigma_b = self.batch_size * self.fraction_noise_std
                bit_pld = _native.gaussian_pld(
                    2.0 * sigma_b, native_cfg
                )
                return inner_pld.compose(bit_pld)


def adaclip(
    inner: _Inner,
    *,
    fraction_noise_std: float = 0.05,
    batch_size: float,
) -> AdaClip:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the observed
    fraction of clipped gradients.  The fraction estimate is made private by
    adding Gaussian noise, which costs a small additional privacy budget.

    For a Gaussian ``inner``, the total privacy cost uses the combined
    sensitivity formula (Theorem 1):

    .. math::

        z_{\\text{eff}} = \\frac{1}{\\sqrt{1/z^2 + 1/(4\\,\\sigma_b^2)}}

    where *z* is the base noise multiplier and
    *σ_b = batch_size × fraction_noise_std*.

    For rectified / truncated Gaussian ``inner``, the inner PLD and the
    bit PLD are composed independently (valid but conservative).

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
        inner: The base mechanism -- gaussian(), rectified_gaussian(),
            or truncated_gaussian().
        fraction_noise_std: Noise std on the clipping *fraction*
            (value in [0, 1]).  Andrew et al. recommend 1/20 = 0.05,
            corresponding to σ_b = m/20 on the clipped-count sum.
        batch_size: Number of examples per training step.  Used to convert
            the fraction-level std to the absolute σ_b needed by
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
    if not isinstance(inner, (Gaussian, RectifiedGaussian, TruncatedGaussian)):
        raise TypeError(
            f"adaclip() requires a Gaussian, RectifiedGaussian, or "
            f"TruncatedGaussian inner mechanism, got {type(inner).__name__}."
        )
    if fraction_noise_std <= 0:
        raise ValueError(
            "fraction_noise_std must be positive, "
            f"got {fraction_noise_std}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return AdaClip(
        inner=inner,
        fraction_noise_std=fraction_noise_std,
        batch_size=batch_size,
    )
