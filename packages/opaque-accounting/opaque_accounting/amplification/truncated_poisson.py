"""Truncated Poisson-subsampled Gaussian mechanism — production DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.transformations.adaclip import AdaClip

#: Mechanism types accepted by :func:`truncated_poisson`.
_Inner = Gaussian | AdaClip


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
        from opaque_accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        match self.inner:
            case Gaussian(noise_multiplier=nm):
                return _native.truncated_poisson_gaussian_pld(
                    nm,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    config.to_native(),
                )
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return _native.truncated_poisson_gaussian_pld(
                    z_eff,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    config.to_native(),
                )
            case _:
                raise TypeError(
                    "TruncatedPoisson requires a Gaussian or AdaClip inner "
                    f"mechanism, got {type(self.inner).__name__}."
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
        inner: The base Gaussian mechanism (from :func:`gaussian`) or
            an :func:`adaclip` transform applied to a Gaussian.
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
    from opaque_accounting.mechanisms.mip_gaussian import MipGaussian

    if isinstance(inner, MipGaussian) or (
        isinstance(inner, AdaClip) and isinstance(inner.inner, MipGaussian)
    ):
        raise TypeError(
            "truncated_poisson() does not support MipGaussian. "
            "Use standard Poisson subsampling instead: "
            "acc.poisson(acc.mip_gaussian(...), rate)"
        )
    if not isinstance(inner, (Gaussian, AdaClip)):
        raise TypeError(
            f"truncated_poisson() requires a Gaussian or AdaClip inner "
            f"mechanism, got {type(inner).__name__}."
        )
    return TruncatedPoisson(
        inner=inner,
        sample_rate=sample_rate,
        batch_size_cap=batch_size_cap,
        dataset_size=dataset_size,
    )
