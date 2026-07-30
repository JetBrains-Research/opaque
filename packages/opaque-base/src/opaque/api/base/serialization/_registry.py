"""Type-keyed serializer registry.

Each opaque wheel registers handlers for the concrete types it owns:

- ``opaque-engine`` registers ``torch.Tensor``, ``numpy.ndarray``.
- ``opaque-accounting`` registers ``Accountant`` and every
  ``DpProcess`` subclass via ``__init_subclass__``.
- ``opaque-dpsgd`` / ``opaque-dpftrl`` register stack-specific state
  classes (e.g. clipping / noise state objects).

The registry holds an exact-type mapping; subclass dispatch is handled
by the dispatcher (see ``_dispatch.py``) when needed.
"""

from __future__ import annotations

from typing import Any

from ._types import FromStateDictFn, StateDictFn

_REGISTRY: dict[type[Any], tuple[StateDictFn, FromStateDictFn]] = {}


def register_serializer(
    typ: type[Any],
    state_dict_fn: StateDictFn,
    from_state_dict_fn: FromStateDictFn,
) -> None:
    """Register a serializer pair for ``typ``.

    Re-registering the same type silently replaces the existing pair —
    this lets test code stub out registrations without unregistering
    first. Production code should not register the same type twice.
    """
    _REGISTRY[typ] = (state_dict_fn, from_state_dict_fn)


def lookup_serializer(
    typ: type[Any],
) -> tuple[StateDictFn, FromStateDictFn] | None:
    """Return the registered serializer pair for ``typ`` or ``None``."""
    return _REGISTRY.get(typ)


__all__ = [
    "lookup_serializer",
    "register_serializer",
]
