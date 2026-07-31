"""Serialization registry, dispatcher, and contract types.

This is the foundation seam every other ``opaque-*`` wheel uses: each
wheel registers its concrete leaf types with the registry on import.
The dispatcher resolves the registry along each node's ``__mro__``, then
falls back to a generic Python container walker (dataclass / NamedTuple
/ tuple / list / dict / primitives). A leaf that is neither registered
nor structural raises instead of being dropped.

User-facing entry points are re-exported on the ``opaque.serialization``
façade.
"""

from __future__ import annotations

from opaque.api.base.serialization._dispatch import (
    from_state_dict,
    state_dict,
)
from opaque.api.base.serialization._registry import (
    is_inert,
    lookup_serializer,
    register_inert_type,
    register_serializer,
    resolve_serializer,
)
from opaque.api.base.serialization._types import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
)

__all__ = [
    # Dispatcher
    "state_dict",
    "from_state_dict",
    # Registry
    "register_serializer",
    "register_inert_type",
    "lookup_serializer",
    "resolve_serializer",
    "is_inert",
    # Contract types
    "Serializer",
    "SerializedState",
    "StateDictFn",
    "FromStateDictFn",
]
