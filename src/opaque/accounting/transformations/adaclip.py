"""Adaptive clipping transformation (Andrew et al. 2021)."""

from __future__ import annotations

import opaque_accounting as _native

from opaque.accounting.mechanisms.gaussian import Gaussian


def adaclip(
    inner: Gaussian,
    quantile_noise_std: float,
) -> Gaussian:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms. The quantile estimation uses a noisy mechanism,
    adding extra privacy cost.

    The total privacy cost uses the combined sensitivity formula:
    ``z_eff = 1 / sqrt(1/z² + 1/(4·σ_b²))``

    where z is the base noise multiplier and σ_b is the quantile noise std.

    The result is a Gaussian mechanism with the effective noise multiplier,
    so it can be composed with :func:`poisson` or :func:`truncated_poisson`::

        step = acc.poisson(acc.adaclip(acc.gaussian(1.1), 50.0), 0.01)

    Note:
        **BoundedGaussian is not currently supported** as the inner mechanism.
        The combined sensitivity formula above was derived for the standard Gaussian
        under Add/Remove adjacency (gradient sensitivity = 1).  For the Bounded
        Gaussian the truncation adds a log-normalisation correction term to the
        privacy loss, and an exact accounting formula is not yet available.
        The needed research is:

        1. Derive the combined sensitivity formula for the bounded Gaussian with
           truncation correction.  The bounded Gaussian under Add/Remove adjacency
           has the same approximate PLD as the standard Gaussian for wide bounds,
           but the interaction with the binary quantile estimator requires careful
           treatment.
        2. Determine how to compute the effective noise multiplier for the combined
           (gradient + quantile) mechanism to plug into
           :func:`poisson` / :func:`truncated_poisson`.
        3. Verify the approximation error introduced by ignoring the log-normalisation
           correction in typical DP-SGD settings.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        quantile_noise_std: Noise std for quantile estimation.
            Larger = more private quantile, less accurate clipping.

    Returns:
        A :class:`Gaussian` with the effective (reduced) noise multiplier.

    Example::

        step = acc.adaclip(acc.gaussian(1.1), quantile_noise_std=50.0)
        eps = step.epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"adaclip() requires a Gaussian inner mechanism, got {type(inner).__name__}."
        )
    # Compute effective noise multiplier: z_eff = 1 / combined_sensitivity
    s = _native.combined_sensitivity(inner.noise_multiplier, quantile_noise_std)
    z_eff = 1.0 / s
    # Return Gaussian so the result can be fed into poisson()/truncated_poisson()
    return Gaussian(z_eff, config=inner.config)
