"""Type definitions for noise operations."""

from abc import ABC

from opaque.random import RngKey


class NoiseState(ABC):
    """Base class for noise state.

    All noise functions (Gaussian and matrix factorization) return a state
    object that inherits from this class, providing a unified interface for
    step tracking and RNG key management.

    Attributes:
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
    """

    _step_counter: int
    """Number of noise_fn calls made."""

    _rng_key: RngKey
    """Immutable RNG key for deterministic per-step derivation."""
