"""Distributed-rank state validation for Gaussian noise.

Registers a :class:`opaque.dpsgd.noise._gaussian.GaussianNoiseState`
sync handler with :func:`opaque.distributed.sync` at import time.
Imported for its side effects from :mod:`opaque.dpsgd.noise`; not
re-exported.
"""

from __future__ import annotations

from opaque.api.dpsgd.noise._gaussian import GaussianNoiseState
from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    register_sync_type,
    sync_object,
)
from opaque.distributed import is_distributed

_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
    "_rng_key": "local",
}


def _assert_rng_key_equal(state: GaussianNoiseState, state_name: str) -> None:
    """Assert that the RNG key seed matches across ranks."""
    assert_scalar_equal(int(state._rng_key.seed), name=f"{state_name}.seed")


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate Gaussian noise state consistency across ranks.

    Asserts that all ranks share the same seed and step counter.  No-op
    outside ``torch.distributed``.
    """
    if not is_distributed():
        return state
    _assert_rng_key_equal(state, "GaussianNoiseState")
    return sync_object(state, field_ops=_NOISE_STATE_FIELD_OPS)


register_sync_type(GaussianNoiseState, sync_gaussian_noise_state)


__all__ = ["sync_gaussian_noise_state"]
