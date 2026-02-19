"""Truncated Poisson-subsampled Gaussian mechanism — production DP-SGD."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class TruncatedPoisson(DpProcess):
    """Truncated Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    batch_size_cap: int
    dataset_size: int
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.truncated_poisson_gaussian_pld(
            self.noise_multiplier,
            self.sample_rate,
            self.batch_size_cap,
            self.dataset_size,
            config=self.config,
        )


def truncated_poisson(
    inner: Gaussian,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> DpProcess:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    In real systems, batch size is capped at ``batch_size_cap`` even though Poisson
    sampling can produce larger batches. This gives tighter privacy bounds than
    standard Poisson subsampling.

    **Use this for production DP-SGD** when you have a fixed batch size limit.

    Also accepts :class:`~opaque.accounting.mechanisms.BoundedGaussian` as ``inner``.
    The bounded Gaussian mechanism (Replace adjacency) has the same PLD as a standard
    Gaussian with ``noise_multiplier / 2``, so
    ``truncated_poisson(bounded_gaussian(nm), ...)`` is equivalent to
    ``truncated_poisson(gaussian(nm / 2), ...)``.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`), or a
            :class:`~opaque.accounting.mechanisms.BoundedGaussian` mechanism
            (from :func:`~opaque.accounting.mechanisms.bounded_gaussian`).
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

        # Also works with bounded Gaussian (Replace adjacency)
        bg = acc.bounded_gaussian(0.8)
        step = acc.truncated_poisson(bg, batch / n, batch, n)
    """
    from opaque.accounting.mechanisms.bounded_gaussian import BoundedGaussian

    if isinstance(inner, BoundedGaussian):
        # BoundedGaussian(nm) has the same PLD as Gaussian(nm/2), so
        # TruncatedPoisson(BoundedGaussian(nm), ...) == TruncatedPoisson(Gaussian(nm/2), ...).
        return TruncatedPoisson(
            inner.noise_multiplier / 2.0,
            sample_rate,
            batch_size_cap,
            dataset_size,
            config=inner.config,
        )
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"truncated_poisson() requires a Gaussian or BoundedGaussian inner mechanism, "
            f"got {type(inner).__name__}."
        )
    return TruncatedPoisson(
        inner.noise_multiplier,
        sample_rate,
        batch_size_cap,
        dataset_size,
        config=inner.config,
    )
