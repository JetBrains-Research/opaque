"""Distributed synchronization helpers for noise components.

These are explicit validation helpers that assert noise state is consistent
across ranks.  Call them manually after each noise step in distributed
training to catch divergence early.
"""

from __future__ import annotations

from opaque.distributed import (
    assert_scalar_equal,
    is_distributed,
    register_sync_type,
    sync_object,
)

from .gaussian_noise import GaussianNoiseState
from .matrix_factorization.noise import MFNoiseState

__all__ = [
    "sync_gaussian_noise_state",
    "sync_mf_noise_state",
]

_NOISE_STATE_OPS = {
    "step_counter": "assert_equal",
}


def _assert_rng_key_equal(state: object, state_name: str) -> None:
    """Assert that the RNG key seed matches across ranks."""
    rng_key = getattr(state, "rng_key", None)
    if rng_key is not None:
        assert_scalar_equal(int(rng_key.seed), name=f"{state_name}.seed")


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate gaussian noise state consistency across ranks.

    Asserts that all ranks have the same seed and step counter.
    No-op if ``torch.distributed`` is not initialized.
    """
    if not is_distributed():
        return state

    _assert_rng_key_equal(state, "GaussianNoiseState")
    return sync_object(state, field_ops=_NOISE_STATE_OPS)


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate MF noise state consistency across ranks.

    Asserts that all ranks have the same seed and step counter.
    No-op if ``torch.distributed`` is not initialized.
    """
    if not is_distributed():
        return state

    _assert_rng_key_equal(state, "MFNoiseState")
    return sync_object(state, field_ops=_NOISE_STATE_OPS)


# Register noise types with the sync dispatcher
register_sync_type(GaussianNoiseState, sync_gaussian_noise_state)
register_sync_type(MFNoiseState, sync_mf_noise_state)
