"""Serialization registry, dispatcher, and contract types.

This is the foundation seam every other ``opaque-*`` wheel uses: each
wheel registers its concrete leaf types with the registry on import.
The dispatcher consults the registry first (exact type, then
``__mro__``), then falls back to a generic Python container walker
(dataclass / NamedTuple / tuple / list / mapping / primitives). A leaf no
handler claims raises ``TypeError``; use
:func:`register_template_restored` to declare one intentionally inert.

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
    register_fallback_resolver,
    register_serializer,
    register_template_restored,
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
    "register_fallback_resolver",
    "register_serializer",
    "register_template_restored",
    "lookup_serializer",
    "resolve_serializer",
    # Contract types
    "Serializer",
    "SerializedState",
    "StateDictFn",
    "FromStateDictFn",
]
