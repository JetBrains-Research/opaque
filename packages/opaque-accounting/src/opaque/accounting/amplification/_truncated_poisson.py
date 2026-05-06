"""Truncated Poisson-subsampled Gaussian mechanism — production DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.mechanisms._gaussian import Gaussian
from opaque.accounting.mechanisms._mf_gaussian import MfGaussian
from opaque.accounting.mechanisms._nonprivate import NonPrivate
from opaque.accounting.transformations._adaclip import AdaClip
from opaque.accounting.transformations._second_moment import SecondMoment

#: Mechanism types accepted by :func:`truncated_poisson`.
_Inner = Gaussian | AdaClip | NonPrivate | SecondMoment


@dataclass(frozen=True, slots=True)
class TruncatedPoisson(DpProcess):
    """Truncated Poisson-subsampled Gaussian mechanism."""

    inner: _Inner
    sample_rate: float
    batch_size_cap: int
    dataset_size: int

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
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                return _native.truncated_poisson_gaussian_pld(
                    nm,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case AdaClip(inner=Gaussian()) as ac:
                # Tight: z_eff combines both into one Gaussian.
                return _native.truncated_poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case SecondMoment(inner=Gaussian(noise_multiplier=0)) | SecondMoment(
                inner=NonPrivate()
            ):
                return _native.non_private_pld(native_cfg)
            case SecondMoment(inner=Gaussian()) as sm:
                # Tight: SecondMoment changes the joint sensitivity, so
                # amplification reduces to truncated Poisson on a Gaussian
                # with effective_nm = σ ÷ joint sensitivity.
                return _native.truncated_poisson_gaussian_pld(
                    sm.noise_multiplier / sm.sensitivity,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "TruncatedPoisson requires a Gaussian, AdaClip(Gaussian), or "
                    "SecondMoment(Gaussian) inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def truncated_poisson(
    inner: _Inner,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> DpProcess:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    In real systems, batch size is capped at ``batch_size_cap`` even though Poisson
    sampling can produce larger batches. This gives tighter privacy bounds than
    standard Poisson subsampling.

    **Use this for production DP-SGD** when you have a fixed batch size limit.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`),
            an :func:`adaclip` transform applied to a Gaussian, or a
            :func:`second_moment` transform applied to a Gaussian.
        sample_rate: Probability of including each example (batch_size / dataset_size).
        batch_size_cap: Maximum batch size (actual batches are capped at this value).
        dataset_size: Total number of examples in the dataset.

    Returns:
        A :class:`TruncatedPoisson` process.

    Example::

        # CIFAR-10: n=50k, batch=250, σ=0.8
        n = 50_000
        batch = 250
        g = acc.gaussian(0.8)
        step = acc.truncated_poisson(g, batch / n, batch, n)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate, SecondMoment)):
        raise TypeError(
            f"truncated_poisson() requires a Gaussian, AdaClip, NonPrivate, or "
            f"SecondMoment(Gaussian) inner mechanism, got {type(inner).__name__}."
        )
    if isinstance(inner, SecondMoment) and isinstance(inner.inner, MfGaussian):
        raise TypeError(
            "truncated_poisson() does not support SecondMoment(MfGaussian) — pass "
            "an MF-aware amplification (cyclic_poisson, b_min_sep) instead."
        )
    return TruncatedPoisson(
        inner=inner,
        sample_rate=sample_rate,
        batch_size_cap=batch_size_cap,
        dataset_size=dataset_size,
    )
