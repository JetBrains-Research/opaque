"""Distributed synchronization helpers for noise components.

These are explicit validation helpers that assert noise state is consistent
across ranks.  Call them manually after each noise step in distributed
training to catch divergence early.
"""

from __future__ import annotations

from opaque.distributed import assert_scalar_equal, is_distributed

from .gaussian_noise import GaussianNoiseState
from .matrix_factorization.noise import MFNoiseState

__all__ = [
    "sync_gaussian_noise_state",
    "sync_mf_noise_state",
]


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate gaussian noise state consistency across ranks.

    Asserts that all ranks have the same seed and step counter.
    No-op if ``torch.distributed`` is not initialized.
    """
    if not is_distributed():
        return state

    assert_scalar_equal(int(state.rng_key.seed), name="GaussianNoiseState.seed")
    assert_scalar_equal(state.step_counter, name="GaussianNoiseState.step_counter")
    return state


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate MF noise state consistency across ranks.

    Asserts that all ranks have the same seed and step counter.
    No-op if ``torch.distributed`` is not initialized.
    """
    if not is_distributed():
        return state

    assert_scalar_equal(int(state.rng_key.seed), name="MFNoiseState.seed")
    assert_scalar_equal(state.step_counter, name="MFNoiseState.step_counter")
    return state
