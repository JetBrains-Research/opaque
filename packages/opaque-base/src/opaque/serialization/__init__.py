"""Functional serialization for Opaque explicit state objects.

Flattens tensor trees, NumPy arrays, dataclasses, named tuples,
sequences, and string-keyed dicts into a flat ``dict[str, Any]``
suitable for ``torch.save`` / ``torch.load``.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf. Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serializer`; registrations also cover subclasses.
``torch.Tensor`` and ``numpy.ndarray`` handlers register automatically
when ``opaque-engine`` is loaded; ``opaque-accounting`` registers PLD
process types; stack wheels register their state objects. Leaves that
genuinely hold nothing worth saving (vendor specs such as
``optree.PyTreeSpec``) are declared with :func:`register_inert_type` and
come back from the template.

Anything else raises :exc:`TypeError` rather than being dropped: a leaf
that vanishes from a state dict restores as whatever the template held,
which for noise, accountant, or optimizer state means a resumed run that
silently diverges from the saved one.
"""

from __future__ import annotations

from opaque.api.base.serialization import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
    from_state_dict,
    is_inert,
    lookup_serializer,
    register_inert_type,
    register_serializer,
    resolve_serializer,
    state_dict,
)

__all__ = [
    "FromStateDictFn",
    "SerializedState",
    "Serializer",
    "StateDictFn",
    "from_state_dict",
    "is_inert",
    "lookup_serializer",
    "register_inert_type",
    "register_serializer",
    "resolve_serializer",
    "state_dict",
]
