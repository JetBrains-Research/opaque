"""Distributed-rank state validation for matrix-factorization noise.

Registers a :class:`opaque.dpftrl.noise._engine.MFNoiseState` sync
handler with :func:`opaque.distributed.sync` at import time.  Imported
for its side effects from :mod:`opaque.dpftrl.noise`; not re-exported.
"""

from __future__ import annotations

from opaque.distributed import is_distributed, register_sync_type, sync_object
from opaque.distributed.state import assert_scalar_equal
from opaque.dpftrl.noise._engine import MFNoiseState


_MF_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
    "_first_max_norm": "assert_equal",
}


def _assert_rng_key_equal(state: MFNoiseState, state_name: str) -> None:
    """Assert that the RNG key seed matches across ranks."""
    assert_scalar_equal(int(state._rng_key.seed), name=f"{state_name}.seed")


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate MF noise state consistency across ranks.

    Asserts that all ranks share the same seed, step counter, and (once
    latched) first-call sensitivity bound.  No-op outside
    ``torch.distributed``.
    """
    if not is_distributed():
        return state
    _assert_rng_key_equal(state, "MFNoiseState")
    return sync_object(state, field_ops=_MF_NOISE_STATE_FIELD_OPS)


register_sync_type(MFNoiseState, sync_mf_noise_state)


__all__ = ["sync_mf_noise_state"]
