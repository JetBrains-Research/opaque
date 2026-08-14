"""Torch distributed registration for backend-neutral optimizer states."""

from __future__ import annotations

import dataclasses
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.distributed._state import (
    assert_pytree_equal,
    assert_scalar_equal,
    register_sync_type,
)
from opaque.api.engine.distributed.collectives import is_distributed
from opaque.api.engine.optimizers import types


def sync_optimizer_state(state: Any) -> Any:
    """Assert that an optimizer state is equal across ranks, preserving it."""
    if not is_distributed():
        return state

    def visit(value: Any, name: str) -> None:
        if ops.is_array(value):
            assert_pytree_equal(value, name=name)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), f"{name}.{field.name}")
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{name}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{name}[{index}]")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            assert_scalar_equal(value, name=name, atol=0.0, rtol=0.0)

    visit(state, type(state).__name__)
    return state


for _state_type in (
    types.AdamState,
    types.SGDState,
    types.LionState,
    types.RAdamState,
    types.RMSpropState,
    types.AdagradState,
    types.AdadeltaState,
    types.AdafactorState,
    types.AdEMAMixState,
    types.ScheduleFreeState,
):
    register_sync_type(_state_type, sync_optimizer_state)


__all__ = ["sync_optimizer_state"]
