"""Per-group Gaussian mechanism — correct accounting for per-group noise.

When per-group clipping is used with independent per-group noise
(``sigma_i = noise_multiplier * C_i`` for each group ``i``), the privacy
loss decomposes as a sum of ``K`` independent Gaussian privacy losses:

.. math::

   L = \\sum_{i=1}^{K} L_i, \\quad L_i \\sim \\text{PLD}_{\\text{Gaussian}}(z)

Because each ``L_i = \\frac{1}{2z^2} + \\frac{Z_i}{z}`` with independent
``Z_i \\sim N(0,1)``:

.. math::

   L = \\frac{K}{2z^2} + \\frac{\\sqrt{K}}{z} Z'
     = \\frac{1}{2(z/\\sqrt{K})^2} + \\frac{1}{z/\\sqrt{K}} Z'

This is **identical** to a single Gaussian mechanism with noise multiplier
``z / sqrt(K)``.  Therefore:

.. math::

   \\text{PLD}_{\\text{Gaussian}(z)}^{\\ast K}
   = \\text{PLD}_{\\text{Gaussian}(z / \\sqrt{K})}

This module provides :func:`per_group_gaussian` which constructs the
correctly-accounted mechanism.
"""

from __future__ import annotations

import math
import warnings

from opaque_accounting.mechanisms.gaussian import Gaussian, gaussian


def per_group_gaussian(noise_multiplier: float, num_groups: int) -> Gaussian:
    """Gaussian mechanism with per-group noise — correct PLD accounting.

    When using per-group clipping with ``K`` groups and independent noise
    ``sigma_i = noise_multiplier * C_i`` per group, the privacy is
    equivalent to a single Gaussian mechanism with effective noise
    multiplier ``noise_multiplier / sqrt(K)``.

    This is because the ``K`` independent Gaussian privacy losses compose
    (convolve in PLD space), and the K-fold self-convolution of
    ``PLD_Gaussian(z)`` equals ``PLD_Gaussian(z / sqrt(K))``.

    Args:
        noise_multiplier: The per-group noise multiplier (sigma / sensitivity
            for each group individually).  This is the same ``z`` used for
            noise generation: ``sigma_i = z * C_i``.
        num_groups: Number of independently-noised groups ``K``.
            When ``K == 1``, returns a standard ``gaussian(noise_multiplier)``.

    Returns:
        A :class:`Gaussian` process with the correctly-adjusted noise
        multiplier for privacy accounting.

    Example::

        # Per-group clipping with 7 groups
        base = per_group_gaussian(1.1, num_groups=7)
        step = acc.poisson(base, sample_rate=0.01)
        eps = (step * 1000).epsilon_at(1e-5)

        # With adaptive clipping (K quantile queries + K noise groups)
        step = acc.poisson(
            acc.adaclip(
                per_group_gaussian(1.1, num_groups=7),
                expected_batch_size=128,
                num_groups=7,
            ),
            sample_rate=0.01,
        )

    Note:
        The ``noise_multiplier`` passed here is the value used for
        **noise generation** (``sigma_i = nm * C_i``).  The privacy
        accounting internally uses ``nm / sqrt(K)``.  Do **not**
        pre-adjust the noise multiplier — this function handles it.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if num_groups == 1:
        return gaussian(noise_multiplier)
    adjusted_nm = noise_multiplier / math.sqrt(num_groups)
    # The adjusted nm can be small even when the per-group nm is reasonable;
    # suppress the low-nm warning since the caller chose a valid per-group nm.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="noise_multiplier=.*is very small")
        return gaussian(adjusted_nm)


__all__ = ["per_group_gaussian"]
