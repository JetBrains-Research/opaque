"""Auto-clip Gaussian mechanism — data-dependent threshold DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class AutoClipGaussian(DpProcess):
    """Auto-clip Gaussian mechanism with data-dependent noise variance.

    Models a Gaussian mechanism where adding/removing one record changes
    both the noiseless output and the noise standard deviation (because
    the clipping threshold is data-dependent).

    The PLD is non-Gaussian when ``noise_ratio != 1`` — it includes a
    chi-squared component from the variance change.
    """

    sensitivity: float
    noise_ratio: float
    dimension: int

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        return _native.auto_clip_gaussian_pld(
            self.sensitivity,
            self.noise_ratio,
            self.dimension,
            config.to_native(),
        )


def auto_clip_gaussian(
    sensitivity: float,
    noise_ratio: float,
    dimension: int,
) -> AutoClipGaussian:
    """Auto-clip Gaussian mechanism for data-dependent threshold DP-SGD.

    Use this when the clipping threshold (and hence noise standard
    deviation) depends on the batch data, as in Auto DP-SGD with
    data-dependent ``C_t``.

    When ``noise_ratio == 1.0``, the PLD is identical to the standard
    Gaussian mechanism with ``noise_multiplier = 1 / sensitivity``.

    Args:
        sensitivity: Worst-case ``||mu(D) - mu(D')|| / v'``, the
            normalized noiseless output change. Analogous to
            ``1 / noise_multiplier`` for the standard Gaussian.
        noise_ratio: ``v(D) / v(D')``, the ratio of noise standard
            deviations under neighboring datasets. Must be in [0.5, 2.0].
            Values close to 1.0 mean the threshold barely changes.
        dimension: Parameter dimension ``d``.

    Returns:
        An :class:`AutoClipGaussian` process.

    Example::

        import opaque.accounting as acc

        # Worst-case parameters from safety-clip analysis
        step = acc.poisson(
            acc.auto_clip_gaussian(
                sensitivity=1.25,   # ||delta_mu|| / v'
                noise_ratio=1.02,   # v(D) / v(D') ≈ 1 + O(1/batch_size)
                dimension=100_000,
            ),
            sample_rate=0.01,
        )
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
    if not 0.5 <= noise_ratio <= 2.0:
        raise ValueError(f"noise_ratio must be in [0.5, 2.0], got {noise_ratio}")
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, got {dimension}")
    return AutoClipGaussian(
        sensitivity=sensitivity,
        noise_ratio=noise_ratio,
        dimension=dimension,
    )
