"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.poisson_gaussian_pld(
            self.noise_multiplier, self.sample_rate, config=self.config
        )


def poisson(
    inner: Gaussian,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise, and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Also accepts :class:`~opaque.accounting.mechanisms.BoundedGaussian` as ``inner``.
    The bounded Gaussian mechanism (Replace adjacency) has the same PLD as a standard
    Gaussian with ``noise_multiplier / 2``, so ``poisson(bounded_gaussian(nm), q)``
    is equivalent to ``poisson(gaussian(nm / 2), q)``.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`), or a
            :class:`~opaque.accounting.mechanisms.BoundedGaussian` mechanism
            (from :func:`~opaque.accounting.mechanisms.bounded_gaussian`).
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        # One training step with standard Gaussian
        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

        # 1000 steps of training
        training = step * 1000
        eps = training.epsilon_at(1e-5)

        # Also works with bounded Gaussian (Replace adjacency)
        step = acc.poisson(acc.bounded_gaussian(1.1), sample_rate=0.01)
    """
    from opaque.accounting.mechanisms.bounded_gaussian import BoundedGaussian

    if isinstance(inner, BoundedGaussian):
        # BoundedGaussian(nm) has the same PLD as Gaussian(nm/2), so
        # Poisson(BoundedGaussian(nm), q) == Poisson(Gaussian(nm/2), q).
        return Poisson(inner.noise_multiplier / 2.0, sample_rate, config=inner.config)
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"poisson() requires a Gaussian or BoundedGaussian inner mechanism, "
            f"got {type(inner).__name__}. "
            "Use: acc.poisson(acc.gaussian(noise_multiplier), sample_rate)"
        )
    return Poisson(inner.noise_multiplier, sample_rate, config=inner.config)
