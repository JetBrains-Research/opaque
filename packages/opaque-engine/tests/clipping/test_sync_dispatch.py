"""``sync()`` dispatch: MRO resolution, marker states, unresolved types.

AUTO-S clipping returns ``AutoClipState`` / ``AutoClippedGradAux``, which
add no fields to their bases. Exact-type dispatch rejected them outright,
so the documented ``sync(clip_state, aux)`` call raised in every AUTO-S
DDP run. These tests need no process group: dispatch happens before any
handler consults the world size.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from opaque.api.engine.clipping._auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)
from opaque.api.engine.clipping._clipped_fun import ClippedFunAux, FixedClipState
from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
from opaque.api.engine.clipping._distributed import sync_marker_clip_state
from opaque.api.engine.types import ClipState
from opaque.distributed import sync


@pytest.mark.parametrize(
    "state",
    [
        FixedClipState(),
        AutoClipState(),
        ClippedFunAux(batch_size=4),
        ClippedGradAux(batch_size=4),
        AutoClippedFunAux(batch_size=4),
        AutoClippedGradAux(batch_size=4),
    ],
    ids=lambda s: type(s).__name__,
)
def test_sync_dispatches_clipping_types(state) -> None:
    synced = sync(state)
    assert type(synced) is type(state)


def test_sync_returns_a_tuple_for_multiple_states() -> None:
    synced = sync(AutoClipState(), AutoClippedGradAux(batch_size=2))
    assert [type(s) for s in synced] == [AutoClipState, AutoClippedGradAux]


def test_sync_raises_for_unresolved_type() -> None:
    @dataclass(frozen=True)
    class _Unregistered:
        value: int = 0

    with pytest.raises(TypeError, match="No sync function registered"):
        sync(_Unregistered())


def test_marker_sync_rejects_a_state_that_carries_fields() -> None:
    """A stateful subclass must register its own handler, not inherit one."""

    @dataclass(frozen=True)
    class _ThresholdClipState(ClipState):
        threshold: float = 1.0

    with pytest.raises(TypeError, match="carries fields"):
        sync_marker_clip_state(_ThresholdClipState())


def test_marker_sync_rejects_a_non_clip_state() -> None:
    with pytest.raises(TypeError, match="Expected a ClipState"):
        sync_marker_clip_state(ClippedGradAux(batch_size=1))
