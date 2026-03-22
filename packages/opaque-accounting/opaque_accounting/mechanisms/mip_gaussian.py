"""MIP Gaussian mechanism — per-example heterogeneous sensitivities."""

from __future__ import annotations

import functools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class MipGaussian(DpProcess):
    """MIP Gaussian mechanism with per-example sensitivities.

    Stores binned (sensitivity, weight) pairs.  The PLD is a weighted mixture
    of Gaussian PLDs — one per sensitivity bucket.
    """

    noise_multiplier: float
    sensitivities: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        # JSON round-trip deserializes tuples as lists; coerce back so
        # the frozen dataclass stays hashable (required by lru_cache).
        if isinstance(self.sensitivities, list):
            object.__setattr__(self, "sensitivities", tuple(self.sensitivities))
        if isinstance(self.weights, list):
            object.__setattr__(self, "weights", tuple(self.weights))

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
        return _native.mip_gaussian_pld(
            self.noise_multiplier,
            list(self.sensitivities),
            list(self.weights),
            config.to_native(),
        )


def _bin_norms(
    norms: Sequence[float], num_bins: int = 100
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Bin raw per-example norms into (sensitivities, weights) pairs.

    Rounds each norm to a grid with ``num_bins`` evenly spaced buckets
    between 0 and max(norms), then aggregates counts into normalized weights.
    Adjacent bins whose sensitivities differ by less than ``merge_rtol``
    (relative) are merged to reduce the number of components.
    """
    n = len(norms)
    max_norm = max(norms)
    if max_norm == 0.0:
        return (0.0,), (1.0,)

    bin_width = max_norm / num_bins
    counts: Counter[float] = Counter()
    for v in norms:
        # Snap to nearest bin center; clamp minimum to half bin width
        bucket = max(round(v / bin_width) * bin_width, bin_width * 0.5)
        counts[bucket] += 1

    sorted_buckets = sorted(counts.keys())
    sensitivities = tuple(sorted_buckets)
    weights = tuple(counts[b] / n for b in sorted_buckets)
    return sensitivities, weights


def mip_gaussian(
    noise_multiplier: float,
    norms: Sequence[float],
    *,
    num_bins: int = 100,
) -> MipGaussian:
    """MIP Gaussian mechanism with per-example gradient norms.

    The constructor bins raw per-example gradient norms into discrete
    sensitivity buckets with associated weights.  The resulting mechanism
    computes a weighted-mixture PLD that is tighter than worst-case
    Gaussian accounting when most examples have small gradients.

    Args:
        noise_multiplier: Noise standard deviation divided by clipping norm (σ/C).
        norms: Raw per-example gradient norms (one per training example).
        num_bins: Number of bins for discretising norms (default 100).

    Returns:
        A :class:`MipGaussian` process.

    Example::

        norms = [0.1, 0.3, 0.5, 0.7, 1.0] * 200
        proc = acc.mip_gaussian(0.8, norms)
        eps = proc.epsilon_at(1e-5)
    """
    if not isinstance(norms, (list, tuple)):
        norms = list(norms)
    if len(norms) == 0:
        raise ValueError("norms must be non-empty")
    for i, v in enumerate(norms):
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"norms[{i}] must be non-negative and finite, got {v}")
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    sensitivities, weights = _bin_norms(norms, num_bins=num_bins)
    return MipGaussian(
        noise_multiplier=noise_multiplier,
        sensitivities=sensitivities,
        weights=weights,
    )
