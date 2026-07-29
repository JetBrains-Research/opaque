"""Public type definitions for :mod:`opaque.dpsgd.noise`.

Re-exports the Gaussian-noise state for type annotations.
"""

from __future__ import annotations

from opaque.api.dpsgd.noise._gaussian import GaussianNoiseState

__all__ = ["GaussianNoiseState"]
