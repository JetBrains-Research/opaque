"""Truncated Poisson-subsampled Gaussian mechanism — production DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config
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

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        match self.inner:
            case Gaussian(noise_multiplier=nm):
                return CgfPld(_native.cgf_truncated_poisson_gaussian_pld(
                    nm, self.sample_rate, self.batch_size_cap, self.dataset_size
                ))
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return CgfPld(_native.cgf_truncated_poisson_gaussian_pld(
                    z_eff, self.sample_rate, self.batch_size_cap, self.dataset_size
                ))
            case _:
                raise NotImplementedError(
                    f"CGF not available for TruncatedPoisson with "
                    f"{type(self.inner).__name__}"
                )

    def pmf(self, **kwargs: object) -> PmfPld:
        cfg = _make_native_config(**kwargs)
        match self.inner:
            case Gaussian(noise_multiplier=nm):
                return PmfPld(_native.truncated_poisson_gaussian_pld(
                    nm,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    cfg,
                ))
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return PmfPld(_native.truncated_poisson_gaussian_pld(
                    z_eff,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    cfg,
                ))
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

    Args:
        inner: The base Gaussian mechanism or an :func:`adaclip` transform.
        sample_rate: Probability of including each example (batch_size / dataset_size).
        batch_size_cap: Maximum batch size.
        dataset_size: Total number of examples in the dataset.

    Returns:
        A :class:`TruncatedPoisson` process.

    Example::

        n = 50_000
        batch = 250
        step = acc.truncated_poisson(acc.gaussian(0.8), batch / n, batch, n)
        training = step * 1000
        eps = training.pmf().epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip)):
        raise TypeError(
            f"truncated_poisson() requires a Gaussian or AdaClip inner mechanism, "
            f"got {type(inner).__name__}."
        )
    return TruncatedPoisson(
        inner=inner,
        sample_rate=sample_rate,
        batch_size_cap=batch_size_cap,
        dataset_size=dataset_size,
    )
