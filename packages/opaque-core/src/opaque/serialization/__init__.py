"""User-facing serialization façade — re-exports from ``opaque.api.base``.

Functional serialization for Opaque explicit state objects. Flattens
tensor trees, NumPy arrays, dataclasses, named tuples, sequences, and
string-keyed dicts into a flat ``dict[str, Any]`` suitable for
``torch.save`` / ``torch.load``. Non-serialisable leaves (vendor specs,
callables, …) are omitted on save and preserved from the *template*
passed to :func:`from_state_dict`.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf. Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serializer`; the default is a structural walk for
generic Python containers plus the torch / NumPy leaf handlers
registered below.
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

# Side-effect import: registers torch.Tensor and numpy.ndarray handlers
# with the base-side registry. Without this, the engine layer would not
# be able to serialize tensors or arrays through ``state_dict``.
from opaque.serialization import _structural  # noqa: F401

__all__ = [
    "state_dict",
    "from_state_dict",
    "register_serializer",
    "lookup_serializer",
    "Serializer",
    "SerializedState",
    "StateDictFn",
    "FromStateDictFn",
]
