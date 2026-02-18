"""Mechanism dataclass types: frozen, stateless DpProcess subclasses.

Each mechanism stores its parameters as immutable fields.  The PLD
is computed on demand via ``pld()`` — each call recomputes from scratch.
Use :func:`~opaque.accounting.composition.cached` to memoize.

Constructor functions (e.g. ``gaussian()``, ``poisson()``) live in
:mod:`opaque.accounting.mechanisms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, Pld, PldConfig


@dataclass(frozen=True, slots=True)
class Gaussian(DpProcess):
    """Gaussian mechanism — stores noise_multiplier, computes PLD on demand."""

    noise_multiplier: float
    config: PldConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.gaussian_pld(self.noise_multiplier, config=self.config)


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    config: PldConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.poisson_gaussian_pld(
            self.noise_multiplier, self.sample_rate, config=self.config
        )


@dataclass(frozen=True, slots=True)
class TruncatedPoisson(DpProcess):
    """Truncated Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    batch_size_cap: int
    dataset_size: int
    config: PldConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.truncated_poisson_gaussian_pld(
            self.noise_multiplier,
            self.sample_rate,
            self.batch_size_cap,
            self.dataset_size,
            config=self.config,
        )


@dataclass(frozen=True, slots=True)
class Accumulated(DpProcess):
    """Accumulated (microbatched) Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    microbatches: int
    config: PldConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.accumulated_poisson_gaussian_pld(
            self.noise_multiplier,
            self.sample_rate,
            self.microbatches,
            config=self.config,
        )


@dataclass(frozen=True, slots=True)
class EpsDelta(DpProcess):
    """Fixed (ε, δ) mechanism."""

    epsilon: float
    delta: float
    config: PldConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.eps_delta_pld(self.epsilon, self.delta, config=self.config)
