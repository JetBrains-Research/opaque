"""Functional serialization for Opaque explicit state objects.

Flattens tensor trees, NumPy arrays, dataclasses, named tuples,
sequences, and string-keyed dicts into a flat ``dict[str, Any]``
suitable for ``torch.save`` / ``torch.load``. Non-serialisable leaves
(vendor specs, callables, …) are omitted on save and preserved from
the *template* passed to :func:`from_state_dict`.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf. Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serializer`. ``torch.Tensor`` and ``numpy.ndarray``
handlers register automatically when ``opaque-engine`` is loaded;
``opaque-accounting`` registers PLD process types; stack wheels
register their state objects.
"""

from __future__ import annotations

from opaque.api.base.serialization import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
    from_state_dict,
    lookup_serializer,
    register_serializer,
    state_dict,
)

__all__ = [
    "FromStateDictFn",
    "SerializedState",
    "Serializer",
    "StateDictFn",
    "from_state_dict",
    "lookup_serializer",
    "register_serializer",
    "state_dict",
]
