"""Type definitions for noise operations."""

from __future__ import annotations

from abc import ABC

from opaque.core.random import RngKey


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


# ---- Distributed sync helpers (shared across mechanisms) ----

# Field-level ops for ``opaque.distributed.sync_object`` applied to any
# ``NoiseState`` subclass.  All concrete noise mechanisms use the same
# step-counter convention, so this is centralized here.
NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
}


def assert_rng_key_equal(state: NoiseState, state_name: str) -> None:
    """Assert that a ``NoiseState``'s RNG key seed matches across ranks.

    Shared across ``sync_gaussian_noise_state`` (opaque-dpsgd) and
    ``sync_mf_noise_state`` (opaque-mf).
    """
    from opaque.core.distributed import assert_scalar_equal

    assert_scalar_equal(int(state._rng_key.seed), name=f"{state_name}.seed")
