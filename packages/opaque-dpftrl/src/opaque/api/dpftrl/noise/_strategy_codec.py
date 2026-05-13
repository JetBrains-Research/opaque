"""Factory-arg serialisation codec for MF strategies.

Strategies are pure recipes — frozen dataclasses carrying only their
factory arguments.  Serialization dumps every dataclass field by name
under a ``type`` discriminator; deserialization calls the matching
strategy factory with that dict.

Each strategy class registers itself via :func:`register_strategy`,
which indexes the class by name (used by :class:`MfGaussian`'s custom
serializer to resolve a strategy sub-dict back to its class) and
installs a ``state_dict`` / ``from_state_dict`` pair on the universal
:mod:`opaque.serialization` registry so direct ``state_dict(strategy)``
works too.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

#: Strategy-class registry, keyed by class name.  Populated by
#: :func:`register_strategy` at import time of each strategy module.
_STRATEGY_REGISTRY: dict[str, type] = {}

#: ``ClsName → factory_callable`` map.  Populated lazily.
_STRATEGY_FACTORIES: dict[str, Any] = {}


def _resolve_factory(cls_name: str):
    """Return the factory function for the given strategy class name."""
    if cls_name in _STRATEGY_FACTORIES:
        return _STRATEGY_FACTORIES[cls_name]
    from opaque.api.dpftrl import noise as _noise

    factory_name = _factory_name_for(cls_name)
    factory = getattr(_noise, factory_name, None)
    if factory is None:
        raise ValueError(
            f"No factory function {factory_name!r} found for strategy {cls_name!r}"
        )
    _STRATEGY_FACTORIES[cls_name] = factory
    return factory


def _factory_name_for(cls_name: str) -> str:
    """Map ``ClsName`` → ``snake_case_strategy`` factory name."""
    base = cls_name
    if base.endswith("Strategy"):
        base = base[: -len("Strategy")]
    if not base:
        raise ValueError(f"unexpected strategy class name: {cls_name!r}")
    out_chars: list[str] = []
    for i, ch in enumerate(base):
        if ch.isupper():
            if i > 0:
                out_chars.append("_")
            out_chars.append(ch.lower())
        else:
            out_chars.append(ch)
    return "".join(out_chars) + "_strategy"


def _to_wire(value: Any) -> Any:
    """Coerce a single field value to a JSON/state-dict-friendly form.

    Callables (e.g. an :data:`opaque.scheduling.types.Schedule` ``lr_schedule``)
    are recipe inputs that cannot be round-tripped through serialization —
    pass them in fresh when reconstructing the strategy.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if callable(value) and not isinstance(value, type):
        raise TypeError(
            "Cannot serialize a callable strategy field "
            f"(type={type(value).__name__}).  Schedules and other callable "
            "recipe inputs must be re-supplied to the strategy factory "
            "after deserialization."
        )
    return value


def serialize_strategy(s: Any) -> dict[str, Any]:
    """Emit ``{"type": ClsName, **factory_args}`` for the strategy."""
    out: dict[str, Any] = {"type": s.__class__.__name__}
    for f in fields(s):
        out[f.name] = _to_wire(getattr(s, f.name))
    return out


def deserialize_strategy(sd: dict[str, Any]) -> Any:
    """Deserialize a strategy dict by calling its factory."""
    sd = dict(sd)
    t = sd.pop("type")
    cls = _STRATEGY_REGISTRY.get(t)
    if cls is None:
        raise ValueError(
            f"Unknown strategy type: {t!r} (registered: {sorted(_STRATEGY_REGISTRY)!r})"
        )
    factory = _resolve_factory(t)
    return factory(**sd)


def register_strategy(cls: type) -> type:
    """Register a strategy dataclass for serialization."""
    _STRATEGY_REGISTRY[cls.__name__] = cls
    from opaque.serialization import register_serializer

    register_serializer(
        cls,
        lambda obj: serialize_strategy(obj),
        lambda _template, sd: deserialize_strategy(dict(sd)),
    )
    return cls


__all__ = [
    "register_strategy",
    "serialize_strategy",
    "deserialize_strategy",
]
