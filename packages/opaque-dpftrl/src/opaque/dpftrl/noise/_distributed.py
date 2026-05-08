"""Distributed-rank state validation for matrix-factorization noise.

Registers a :class:`opaque.dpftrl.noise._engine.MFNoiseState` sync
handler with :func:`opaque.distributed.sync` at import time.  Imported
for its side effects from :mod:`opaque.dpftrl.noise`; not re-exported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from opaque.distributed import is_distributed
from opaque.distributed._state import (
    assert_scalar_equal,
    register_sync_type,
    sync_object,
)
from opaque.dpftrl.noise._engine import MFNoiseState
from opaque.types import PerGroup


def _fingerprint_per_group(pg: PerGroup) -> float:
    """Deterministic float fingerprint for cross-rank equality of ``PerGroup``."""
    payload = {
        "groups": sorted(pg.groups.items()),
        "values": sorted(pg.values.items()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**53)


def _sync_mf_first_max_norm(value: Any, device: Any = None) -> None:
    """Assert ``_first_max_norm`` matches across ranks (scalar or ``PerGroup``)."""
    if value is None:
        return
    if isinstance(value, PerGroup):
        assert_scalar_equal(
            _fingerprint_per_group(value),
            name="MFNoiseState._first_max_norm(PerGroup fingerprint)",
            device=device,
        )
        return
    assert_scalar_equal(
        float(value), name="MFNoiseState._first_max_norm", device=device
    )


_MF_NOISE_STATE_FIELD_OPS: dict[str, str | Callable[..., Any]] = {
    "_step_counter": "assert_equal",
    "_first_max_norm": _sync_mf_first_max_norm,
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
