"""Backend-neutral adaptive clipping distributed-state behavior."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from opaque.api.dpsgd.clipping import _distributed as adaptive_distributed
from opaque.api.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opaque.random import key

if TYPE_CHECKING:
    import pytest


def test_distributed_sync_requires_a_shared_rng_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive threshold noise must start from the same key on every rank."""
    _, state = adaptive_clipped_grad(
        lambda params, batch: (params * batch).sum(),
        initial_clipping_norm=1.0,
        key=key(37),
        batch_argnums=1,
    )
    calls: list[tuple[int, str]] = []

    monkeypatch.setattr(adaptive_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(
        adaptive_distributed,
        "assert_scalar_equal",
        lambda value, *, name: calls.append((value, name)),
    )
    monkeypatch.setattr(
        adaptive_distributed,
        "sync_object",
        lambda value, *, field_ops: value,
    )

    synced = adaptive_distributed.sync_adaptive_clip_state(state)

    # ``sync`` normalizes the integer counters to the float wire format
    # before the collective; everything else passes through unchanged.
    assert synced == replace(state, _num_clipped=0.0, _batch_size=0.0)
    assert calls == [(37, "AdaptiveClipState.seed")]
