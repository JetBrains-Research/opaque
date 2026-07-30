"""Serialization registry, dispatcher, and contract types.

This is the foundation seam every other ``opaque-*`` wheel uses: each
wheel registers its concrete leaf types with the registry on import.
The dispatcher consults the registry first, then falls back to a
generic Python container walker (dataclass / NamedTuple / tuple / list
/ dict / primitives).

User-facing entry points are re-exported on the ``opaque.serialization``
façade.
"""

from __future__ import annotations

from opaque.api.base.serialization._dispatch import (
    from_state_dict,
    state_dict,
)
from opaque.api.base.serialization._registry import (
    lookup_serializer,
    register_serializer,
)
from opaque.api.base.serialization._types import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
)

__all__ = [
    "FromStateDictFn",
    "SerializedState",
    # Contract types
    "Serializer",
    "StateDictFn",
    "from_state_dict",
    "lookup_serializer",
    # Registry
    "register_serializer",
    # Dispatcher
    "state_dict",
]
