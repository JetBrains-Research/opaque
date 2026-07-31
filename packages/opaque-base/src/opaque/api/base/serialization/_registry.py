"""Type-keyed serializer registry.

Each opaque wheel registers handlers for the concrete types it owns:

- ``opaque-engine`` registers ``torch.Tensor``, ``numpy.ndarray``.
- ``opaque-accounting`` registers ``Accountant`` and every
  ``DpProcess`` subclass via ``__init_subclass__``.
- ``opaque-dpsgd`` / ``opaque-dpftrl`` register stack-specific state
  classes (e.g. clipping / noise state objects).

Two kinds of declaration are possible:

- :func:`register_serializer` — the type owns state that must survive a
  save/load round trip.
- :func:`register_inert_type` — the type carries no state worth saving
  (vendor specs such as ``optree.PyTreeSpec``), so the value from the
  restore template is authoritative.

Registrations are keyed by exact type; :func:`resolve_serializer` and
:func:`is_inert` extend a lookup along ``__mro__`` so a subclass of a
registered type (``torch.nn.Parameter`` for ``torch.Tensor``) resolves
to its nearest declared ancestor. A leaf that resolves to neither is a
dispatch error, not something to skip — see ``_dispatch.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ._types import FromStateDictFn, StateDictFn

_REGISTRY: dict[type[Any], tuple[StateDictFn, FromStateDictFn]] = {}
_INERT: set[type[Any]] = set()

_Resolution = (
    tuple[Literal["serializer"], tuple["StateDictFn", "FromStateDictFn"]]
    | tuple[Literal["inert"], None]
    | None
)

# Memoizes the ``__mro__`` walk. Cleared by every registration so a late
# (or test-stubbed) registration cannot be shadowed by a stale entry.
_RESOLVED: dict[type[Any], _Resolution] = {}


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
    _INERT.discard(typ)
    _RESOLVED.clear()


def register_inert_type(typ: type[Any]) -> None:
    """Declare that ``typ`` holds no state to serialize.

    Instances are omitted from :func:`~opaque.serialization.state_dict`
    and restored from the template by
    :func:`~opaque.serialization.from_state_dict`. Use this only for
    values that are reconstructed with the surrounding state and carry
    no numbers of their own — an ``optree.PyTreeSpec`` describing a tree
    layout, for instance. Anything holding numeric state needs a real
    serializer pair instead, otherwise it silently disappears from every
    checkpoint.
    """
    if typ in _REGISTRY:
        raise ValueError(
            f"{typ.__qualname__} already has a registered serializer; it "
            "cannot also be declared inert."
        )
    _INERT.add(typ)
    _RESOLVED.clear()


def lookup_serializer(
    typ: type[Any],
) -> tuple[StateDictFn, FromStateDictFn] | None:
    """Return the serializer pair registered for exactly ``typ``, or ``None``."""
    return _REGISTRY.get(typ)


def _resolve(typ: type[Any]) -> _Resolution:
    """Resolve ``typ`` to its nearest declared ancestor along ``__mro__``."""
    cached = _RESOLVED.get(typ, ...)
    if cached is not ...:
        return cached  # type: ignore[return-value]

    resolution: _Resolution = None
    for base in typ.__mro__:
        pair = _REGISTRY.get(base)
        if pair is not None:
            resolution = ("serializer", pair)
            break
        if base in _INERT:
            resolution = ("inert", None)
            break

    _RESOLVED[typ] = resolution
    return resolution


def resolve_serializer(
    typ: type[Any],
) -> tuple[StateDictFn, FromStateDictFn] | None:
    """Return the serializer pair for ``typ`` or its nearest registered base.

    Unlike :func:`lookup_serializer` this walks ``__mro__``, so a
    ``torch.Tensor`` subclass resolves to the tensor handler instead of
    falling through to the structural walker. A type whose nearest
    declaration is :func:`register_inert_type` resolves to ``None``.
    """
    resolution = _resolve(typ)
    if resolution is not None and resolution[0] == "serializer":
        return resolution[1]
    return None


def is_inert(typ: type[Any]) -> bool:
    """Whether ``typ`` (or its nearest declared base) is registered inert."""
    resolution = _resolve(typ)
    return resolution is not None and resolution[0] == "inert"


__all__ = [
    "is_inert",
    "lookup_serializer",
    "register_inert_type",
    "register_serializer",
    "resolve_serializer",
]
