"""Type-keyed serializer registry.

Each opaque wheel registers handlers for the concrete types it owns:

- ``opaque-engine`` registers ``torch.Tensor``, ``numpy.ndarray``.
- ``opaque-accounting`` registers ``Accountant`` and every
  ``DpProcess`` subclass via ``__init_subclass__``.
- ``opaque-dpsgd`` / ``opaque-dpftrl`` register stack-specific state
  classes (e.g. clipping / noise state objects).

The registry holds an exact-type mapping; subclass dispatch is handled
by :func:`resolve_serializer`, which walks ``__mro__`` so a subclass
(``nn.Parameter`` against a registered ``torch.Tensor``) reaches its
base class handler instead of falling through to the generic walker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def register_template_restored(typ: type[Any]) -> None:
    """Declare ``typ`` inert: nothing is saved, the template supplies it.

    The escape hatch for leaves that carry no run state and are fully
    determined by a freshly-built template — vendor structure handles
    such as ``optree.PyTreeSpec``, for example. Without a registration
    the dispatcher raises on an unrecognised non-container leaf rather
    than dropping it, so this is how a wheel says "dropping this one is
    intentional".

    Do not use it for anything a resumed run needs to reproduce: the
    saved checkpoint will not contain the value, and a template built
    with different settings restores those different settings.
    """
    register_serializer(typ, lambda _obj: {}, lambda template, _sd: template)


def lookup_serializer(
    typ: type[Any],
) -> tuple[StateDictFn, FromStateDictFn] | None:
    """Return the serializer pair registered for exactly ``typ``, or ``None``."""
    return _REGISTRY.get(typ)


def resolve_serializer(
    typ: type[Any],
) -> tuple[StateDictFn, FromStateDictFn] | None:
    """Return the serializer pair for ``typ`` or its nearest registered base.

    Exact type first, then ``__mro__`` order, so ``nn.Parameter`` and any
    other ``torch.Tensor`` subclass resolve to the ``torch.Tensor``
    handler.  Returns ``None`` when nothing in the MRO is registered.
    """
    pair = _REGISTRY.get(typ)
    if pair is not None:
        return pair
    for base in typ.__mro__[1:]:
        pair = _REGISTRY.get(base)
        if pair is not None:
            return pair
    return None


_FALLBACK_RESOLVERS: list[Any] = []


def register_fallback_resolver(resolver: Any) -> None:
    """Register a last-chance hook consulted for unrecognised leaf values.

    A resolver receives the offending value and returns ``True`` when it
    made progress (typically by importing a provider whose activation
    registers native-type serializers), prompting one dispatch retry.
    Engine/provider wheels install these so ``state_dict`` /
    ``from_state_dict`` work on native arrays before any explicit backend
    activation.
    """
    if not callable(resolver):
        raise TypeError("resolver must be callable")
    _FALLBACK_RESOLVERS.append(resolver)


def run_fallback_resolvers(value: Any) -> bool:
    """Invoke registered fallback resolvers; True when any made progress."""
    progressed = False
    for resolver in _FALLBACK_RESOLVERS:
        if resolver(value):
            progressed = True
    return progressed


__all__ = [
    "lookup_serializer",
    "register_fallback_resolver",
    "register_serializer",
    "register_template_restored",
    "resolve_serializer",
]
