"""Private second-moment transformation for Gaussian / MF Gaussian mechanisms.

When a training loop releases both noised gradients and noised element-wise
squared gradients, the first stream needs a larger sensitivity.  Under
Opaque's add/remove adjacency model the default d >= 2 overhead is
``sqrt(3/2)`` over the first-moment-only stream.

The transformation accepts either a :class:`Gaussian` (DP-SGD baseline),
an :class:`AdaClip` over Gaussian (DP-SGD with adaptive clipping), or
:class:`MfGaussian` (DP-FTRL with correlated noise) inner mechanism.
For Gaussian / AdaClip(Gaussian) inner the c1 / c2 max column norms
collapse to 1.0 (identity strategy); for MfGaussian they come from the
strategy's encoder matrix.

For the :class:`AdaClip` inner the per-step Gaussian first folds the
quantile-estimator noise into the gradient noise via Theorem 1's
``z_eff`` (giving an effective single-Gaussian noise multiplier), and
the second-moment overhead is then applied on that effective Gaussian.
This composition is valid because the threshold-quantile and
gradient/squared-gradient releases use independent randomness.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from .. import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.mechanisms._gaussian import Gaussian
from opaque.accounting.mechanisms._mf_gaussian import MfGaussian
from opaque.accounting.mechanisms._nonprivate import NonPrivate
from opaque.accounting.transformations._adaclip import AdaClip


_DEFAULT_SECOND_MOMENT_OVERHEAD = math.sqrt(3.0 / 2.0)
_Inner = Gaussian | MfGaussian | AdaClip


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

    inner: _Inner
    input_sensitivity: float
    max_column_norm: float | None = None
    first_moment_overhead: float = _DEFAULT_SECOND_MOMENT_OVERHEAD

    @property
    def noise_multiplier(self) -> float:
        """Single-Gaussian noise multiplier of the underlying release.

        :class:`Gaussian` / :class:`MfGaussian` inners surface their own raw
        ``noise_multiplier``; :class:`AdaClip` inner returns the
        :attr:`AdaClip.effective_noise_multiplier` that folds the
        quantile-estimator privacy cost into the gradient noise so the
        joint PLD reduces to a single-Gaussian computation.
        """
        match self.inner:
            case AdaClip() as ac:
                return ac.effective_noise_multiplier
            case _:
                return self.inner.noise_multiplier

    @property
    def _c1_norm(self) -> float:
        """Max column norm of the first-moment strategy matrix.

        For :class:`MfGaussian` inner this is the encoder's column-norm
        (or an explicit override).  For :class:`Gaussian` (and
        :class:`AdaClip` over Gaussian) inner the identity strategy has
        unit columns, so c1 = 1.0.
        """
        if self.max_column_norm is not None:
            return self.max_column_norm
        match self.inner:
            case MfGaussian() as mf:
                return mf.sensitivity
            case _:
                return 1.0

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
        native_cfg = config.to_native()
        match self.inner:
            case Gaussian(noise_multiplier=0) | AdaClip(inner=NonPrivate()):
                return _native.non_private_pld(native_cfg)
            case AdaClip(inner=Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case Gaussian() | AdaClip(inner=Gaussian()):
                # Effective noise multiplier for the first stream: σ ÷ joint
                # sensitivity = σ ÷ (input_sensitivity · c1 · overhead).  For
                # AdaClip inner, ``self.noise_multiplier`` returns the
                # z_eff-folded value that already encodes the quantile-
                # estimator privacy cost.
                return _native.gaussian_pld(
                    self.noise_multiplier / self.sensitivity,
                    native_cfg,
                )
            case MfGaussian():
                return _native.mf_gaussian_pld(
                    self.noise_multiplier,
                    self.sensitivity,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "SecondMoment.pld requires a Gaussian, AdaClip(Gaussian), "
                    f"or MfGaussian inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def second_moment(
    inner: _Inner,
    *,
    sensitivity: float,
    max_column_norm: float | None = None,
    first_moment_overhead: float = _DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> SecondMoment:
    """Account for releasing private squared gradients alongside gradients.

    Args:
        inner: A Gaussian-family mechanism — :func:`gaussian` for DP-SGD,
            :func:`adaclip` over Gaussian for DP-SGD with adaptive
            clipping, or any MF Gaussian (``band_mf()``, ``blt()``,
            ``lambda_cgd()``, ``bisr()``, ``bsr()``) for DP-FTRL.
        sensitivity: Clipped-gradient sensitivity before the strategy is
            applied.  For averaged gradients this is typically
            ``clipping_norm / batch_size``.
        max_column_norm: Max column norm of the first-moment strategy
            matrix.  If ``None``, defaults to ``inner.sensitivity`` for
            MF inner or ``1.0`` for Gaussian / AdaClip(Gaussian) inner.
        first_moment_overhead: Overhead applied to the first stream.  Defaults
            to ``sqrt(3/2)`` for d >= 2 add/remove DP.

    Returns:
        A :class:`SecondMoment` process.
    """
    match inner:
        case Gaussian() | MfGaussian() | AdaClip(inner=Gaussian()):
            pass
        case AdaClip():
            raise TypeError(
                "second_moment() over AdaClip requires a Gaussian inside the "
                f"AdaClip wrapper, got AdaClip({type(inner.inner).__name__})."
            )
        case _:
            raise TypeError(
                "second_moment() requires a Gaussian mechanism — gaussian(), "
                "adaclip(gaussian(...)), or any MF Gaussian (band_mf, blt, "
                "lambda_cgd, bisr, bsr); "
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
