"""Private second-moment transformation for MF Gaussian mechanisms.

When a training loop releases both noisy gradients and noisy element-wise
squared gradients, the first stream needs a larger sensitivity.  Under
Opaque's add/remove adjacency model the default d >= 2 overhead is
``sqrt(3/2)`` over the first-moment-only stream.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from .. import _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.mf_gaussian import MfGaussian


_DEFAULT_SECOND_MOMENT_OVERHEAD = math.sqrt(3.0 / 2.0)


def _second_moment_joint_sensitivity(
    c1_max_column_norm: float,
    sensitivity: float,
    *,
    first_moment_overhead: float,
) -> float:
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return sensitivity * c1_max_column_norm * first_moment_overhead


@dataclass(frozen=True, slots=True)
class SecondMoment(DpProcess):
    """Transformation for private first+second moment estimation."""

    inner: MfGaussian
    input_sensitivity: float
    max_column_norm: float | None = None
    first_moment_overhead: float = _DEFAULT_SECOND_MOMENT_OVERHEAD

    @property
    def noise_multiplier(self) -> float:
        return self.inner.noise_multiplier

    @property
    def _c1_norm(self) -> float:
        """Max column norm of the first-moment strategy matrix."""
        if self.max_column_norm is not None:
            return self.max_column_norm
        return self.inner.sensitivity

    @property
    def sensitivity(self) -> float:
        """Effective first-stream sensitivity for private second moments."""
        return _second_moment_joint_sensitivity(
            self._c1_norm,
            self.input_sensitivity,
            first_moment_overhead=self.first_moment_overhead,
        )

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
        from opaque.accounting.discretization import get_discretization

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


def second_moment(
    inner: MfGaussian,
    *,
    sensitivity: float,
    max_column_norm: float | None = None,
    first_moment_overhead: float = _DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> SecondMoment:
    """Account for releasing private squared gradients alongside gradients.

    Args:
        inner: Any MF Gaussian mechanism, such as ``band_mf()``, ``blt()``,
            ``lambda_cgd()``, ``bisr()``, or ``bsr()``.
        sensitivity: Clipped-gradient sensitivity before the MF strategy is
            applied.  For averaged gradients this is typically
            ``clipping_norm / batch_size``.
        max_column_norm: Max column norm of the first strategy matrix.  If
            ``None``, falls back to ``inner.sensitivity``.
        first_moment_overhead: Overhead applied to the first stream.  Defaults
            to ``sqrt(3/2)`` for d >= 2 add/remove DP.

    Returns:
        A :class:`SecondMoment` process.
    """
    if not isinstance(inner, MfGaussian):
        raise TypeError(
            "second_moment() requires an MfGaussian mechanism "
            "(band_mf, blt, lambda_cgd, bisr, bsr), "
            f"got {type(inner).__name__}."
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return SecondMoment(
        inner=inner,
        input_sensitivity=sensitivity,
        max_column_norm=max_column_norm,
        first_moment_overhead=first_moment_overhead,
    )


__all__ = ["SecondMoment", "second_moment"]
