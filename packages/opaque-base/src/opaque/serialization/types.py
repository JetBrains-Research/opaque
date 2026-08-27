"""Serialization types — the serializer protocol and its callable aliases.

:func:`opaque.serialization.register_serializer` consumes them and
:func:`opaque.serialization.lookup_serializer` returns one; they live
here for type annotations, matching :mod:`opaque.optimizers.types`.
"""

from opaque.api.base.serialization import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
)

__all__ = [
    "FromStateDictFn",
    "SerializedState",
    "Serializer",
    "StateDictFn",
]
