"""Distributed synchronization helpers for noise components."""

from __future__ import annotations

from dataclasses import replace

from opaque.distributed import assert_scalar_equal, is_distributed

from .gaussian_noise import GaussianNoiseState
from .matrix_factorization.noise import MFNoiseState

__all__ = [
    "sync_gaussian_noise_state",
    "sync_mf_noise_state",
]


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate and synchronize gaussian noise state across ranks.

    For synchronized mode, all ranks must keep identical step counters and
    canonical seeds.
    """
    if not is_distributed() or not state.synchronized:
        return state

    assert_scalar_equal(state.seed, name="GaussianNoiseState.seed")
    assert_scalar_equal(state.step_counter, name="GaussianNoiseState.step_counter")
    return state


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate and synchronize MF noise state across ranks.

    For synchronized mode, all ranks must keep identical outer counters.
    """
    if not is_distributed() or not state.synchronized:
        return state

    assert_scalar_equal(state.seed, name="MFNoiseState.seed")
    assert_scalar_equal(state.step_counter, name="MFNoiseState.step_counter")
    return replace(state)
