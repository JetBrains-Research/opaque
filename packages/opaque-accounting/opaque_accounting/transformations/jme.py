"""JME transformation — accounts for joint first+second moment estimation.

When using JME (arXiv:2502.06597) to enable DP-Adam, the noise must
cover both moment streams.  Under add/remove DP (Opaque's model), the
joint sensitivity is ``ζ · ‖C₁‖_{1→2} · √(1 + 1/c_d)`` where
``‖C₁‖_{1→2}`` is the strategy's max column norm and ``ζ`` is the
per-sample clipping bound.  For d ≥ 2 this is ``ζ · S · √(3/2)``
— the second moment costs ~22% more noise than first-moment-only.

Usage::

    # SGD (no JME):
    acc.cyclic_poisson(acc.band_mf(nm, sensitivity=S, num_groups=k), sample_rate=q)

    # Adam (with JME):
    acc.cyclic_poisson(acc.jme(acc.band_mf(nm, sensitivity=S, num_groups=k), zeta=ζ), sample_rate=q)

References:
    - Kalinin, Upadhyay, Lampert (2025) "Continual Release Moment
      Estimation with Differential Privacy" https://arxiv.org/abs/2502.06597
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.mf_gaussian import MfGaussian


@dataclass(frozen=True, slots=True)
class Jme(DpProcess):
    """JME transformation — joint first+second moment estimation.

    Wraps an ``MfGaussian`` mechanism and adjusts the sensitivity to
    account for privately estimating both moments under add/remove DP.
    The effective sensitivity becomes
    ``zeta × max_column_norm × √(1 + 1/c_d)`` (Theorem 3.2 adapted
    to add/remove adjacency).

    Amplification modules (``cyclic_poisson``, ``balls_in_bins``)
    pattern-match on ``Jme`` and unwrap the inner mechanism with the
    adjusted sensitivity, just like ``AdaClip``.
    """

    inner: MfGaussian
    zeta: float
    max_column_norm: float | None = None

    @property
    def noise_multiplier(self) -> float:
        return self.inner.noise_multiplier

    @property
    def _c1_norm(self) -> float:
        """‖C₁‖_{1→2} — max column norm of the strategy matrix."""
        if self.max_column_norm is not None:
            return self.max_column_norm
        return self.inner.sensitivity

    @property
    def sensitivity(self) -> float:
        """Joint sensitivity under add/remove DP.

        ``s = ζ · ‖C₁‖_{1→2} · √(1 + 1/c_d)``

        For d ≥ 2: ``s = ζ · ‖C₁‖ · √(3/2)``.
        """
        import math

        cd = 2.0  # c_d for d ≥ 2 (neural networks)
        return self.zeta * self._c1_norm * math.sqrt(1.0 + 1.0 / cd)

    @property
    def gram_matrix(self) -> tuple[float, ...] | None:
        return getattr(self.inner, "gram_matrix", None)

    @property
    def num_groups(self) -> int:
        return getattr(self.inner, "num_groups", 1)

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
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity,
            config.to_native(),
        )


def jme(
    inner: MfGaussian,
    *,
    zeta: float,
    max_column_norm: float | None = None,
) -> Jme:
    """Account for JME joint moment estimation.

    Wraps an MF mechanism to reflect the privacy cost of estimating both
    first and second moments.  The joint sensitivity (Theorem 3.2 of
    arXiv:2502.06597) is ``2ζ · ‖C₁‖_{1→2}``.

    Args:
        inner: Any MF mechanism — ``band_mf()``, ``blt()``,
            ``lambda_cgd()``, ``bisr()``.
        zeta: Per-sample clipping bound (``clip_state.sensitivity``,
            typically ``clipping_norm / batch_size``).
        max_column_norm: Max column norm ``‖C₁‖_{1→2}`` of the strategy
            matrix.  If ``None``, falls back to ``inner.sensitivity``
            (correct for single-participation; conservative for
            multi-participation).  Pass ``strategy._max_column_norm``
            for tight multi-participation accounting.

    Returns:
        A :class:`Jme` process.

    Example::

        def acct(nm):
            return acc.cyclic_poisson(
                acc.jme(acc.band_mf(nm, sensitivity=S, num_groups=k), zeta=zeta),
                sample_rate=q,
            )
        result = cal.calibrate(budget, acct, param_min=0.1, param_max=20.0)
    """
    if not isinstance(inner, MfGaussian):
        raise TypeError(
            f"jme() requires an MfGaussian mechanism (band_mf, blt, lambda_cgd, bisr), "
            f"got {type(inner).__name__}."
        )
    if zeta <= 0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    return Jme(inner=inner, zeta=zeta, max_column_norm=max_column_norm)
