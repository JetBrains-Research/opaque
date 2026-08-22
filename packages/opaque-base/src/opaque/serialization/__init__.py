"""Functional serialization for Opaque explicit state objects.

Flattens native array trees, dataclasses, named tuples, sequences, and
string-keyed dicts into a flat ``dict[str, Any]`` suitable for a
provider's checkpoint storage.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf. Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serializer`. Lookup is by exact type and then by
``__mro__``, so a subclass of a registered type (a custom
``torch.Tensor`` subclass, say) uses the nearest base class handler.
``numpy.ndarray`` support loads with ``opaque-engine``; a provider
registers its native array handlers when it activates —
``opaque-torch`` covers ``torch.Tensor`` and gives ``torch.nn.Parameter``
an exact-type handler (preserving the subclass and ``requires_grad``)
rather than relying on the ``__mro__`` fallback. ``opaque-accounting`` registers PLD process types; stack
wheels register their state objects.

A leaf that is neither registered nor a generic container nor a
primitive raises ``TypeError`` on both save and restore rather than
being dropped. Declare the genuinely inert ones — vendor structure
handles and the like — with :func:`register_template_restored`.

The ``Serializer`` protocol and the ``StateDictFn`` /
``FromStateDictFn`` / ``SerializedState`` aliases a registration is
written against live in :mod:`opaque.serialization.types`.
"""

from __future__ import annotations

from opaque.api.base.serialization import (
    from_state_dict,
    lookup_serializer,
    register_serializer,
    register_template_restored,
    resolve_serializer,
    state_dict,
)
from opaque.serialization import types

__all__ = [
    "from_state_dict",
    "lookup_serializer",
    "register_serializer",
    "register_template_restored",
    "resolve_serializer",
    "state_dict",
    "types",
]
