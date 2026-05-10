"""Distributed-rank state validation for matrix-factorization noise.

Registers a :class:`opaque.dpftrl.noise._engine.MFNoiseState` sync
handler with :func:`opaque.distributed.sync` at import time.  Imported
for its side effects from :mod:`opaque.dpftrl.noise`; not re-exported.
"""

from __future__ import annotations

import hashlib
import json

from opaque.distributed import is_distributed
from opaque.distributed._state import (
    assert_scalar_equal,
    register_sync_type,
    sync_object,
)
from opaque.dpftrl.noise._engine import MFNoiseState
from opaque.types import PerGroup


def fingerprint_scalar_max_norm(c: float) -> int:
    """Deterministic 64-bit fingerprint for a scalar latched ``max_norm``."""
    payload = {"kind": "mf_noise_scalar_max_norm", "c": float(c)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def fingerprint_per_group_max_norm(pg: PerGroup) -> int:
    """Deterministic 64-bit integer fingerprint for cross-rank equality of
    ``PerGroup``.

    Called once when ``mf_noise`` latches a ``PerGroup`` ``max_norm``; the
    result is stored on :class:`MFNoiseState` and equality is checked on each
    :func:`sync` (avoids re-hashing the full param→group map on the hot path).

    Scalar and per-group norms use disjoint ``kind`` payloads so a scalar latch
    on one rank cannot collide with a per-group latch fingerprint on another.
    """
    payload = {
        "kind": "mf_noise_per_group_max_norm",
        "groups": sorted(pg.groups.items()),
        "values": sorted(pg.values.items()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


_MF_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
    "_first_max_norm_sync_fingerprint": "assert_equal",
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


__all__ = [
    "fingerprint_per_group_max_norm",
    "fingerprint_scalar_max_norm",
    "sync_mf_noise_state",
]
