"""Factory-arg serialisation codec for MF strategies.

Strategy state_dict carries the **public** factory args only (the user's
configured identity of the strategy: ``n_steps``, ``bands``, ``min_sep``,
``momentum``, etc.).  The codec drops every ``_``-prefixed field — those
are internal derivations that the factory recomputes on load.

Each strategy class registers itself via :func:`register_strategy`, which
records the class in a name index used by ``MfGaussian``'s custom
serializer and installs a ``state_dict`` / ``from_state_dict`` pair on
the universal :mod:`opaque.serialization` registry so direct
``state_dict(strategy)`` works too.

On load, the codec calls the strategy factory with the deserialized
public fields, which recomputes ``sensitivity``, ``_coefficients``,
``_gram_matrix``, ``_streaming_matrix``, etc.  Round-trip equality holds
when the factory is deterministic (closed-form for BSR / BISR / λCGD /
Band; BLT depends on L-BFGS being numerically reproducible on the same
machine).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

#: Strategy-class registry, keyed by class name.  Populated by
#: :func:`register_strategy` at import time of each strategy module.
_STRATEGY_REGISTRY: dict[str, type] = {}

#: ``ClsName → factory_callable`` map.  Populated lazily.
_STRATEGY_FACTORIES: dict[str, Any] = {}

_PRIMITIVE_TYPES = (int, float, bool, str, type(None))


def _resolve_factory(cls_name: str):
    """Return the factory function for the given strategy class name."""
    if cls_name in _STRATEGY_FACTORIES:
        return _STRATEGY_FACTORIES[cls_name]
    from opaque.api.dpftrl import noise as _noise

    factory_name = _factory_name_for(cls_name)
    factory = getattr(_noise, factory_name, None)
    if factory is None:
        raise ValueError(
            f"No factory function {factory_name!r} found for strategy "
            f"{cls_name!r}"
        )
    _STRATEGY_FACTORIES[cls_name] = factory
    return factory


def _factory_name_for(cls_name: str) -> str:
    """Map ``ClsName`` (e.g. ``"BltStrategy"``) → factory name (``"blt_strategy"``)."""
    # IdentityStrategy → identity_strategy
    # BandMfStrategy → band_mf_strategy
    # BltStrategy → blt_strategy, etc.
    base = cls_name
    if base.endswith("Strategy"):
        base = base[: -len("Strategy")]
    if not base:
        raise ValueError(f"unexpected strategy class name: {cls_name!r}")
    # CamelCase → snake_case
    out_chars: list[str] = []
    for i, ch in enumerate(base):
        if ch.isupper():
            if i > 0:
                out_chars.append("_")
            out_chars.append(ch.lower())
        else:
            out_chars.append(ch)
    return "".join(out_chars) + "_strategy"


#: Field names that are computed by the factory and not stored on the wire.
#: ``sensitivity`` is determined by the other factory args; including it would
#: be redundant (and confusing if it drifted from what the factory recomputes).
_COMPUTED_FIELDS: frozenset[str] = frozenset({"sensitivity"})


def serialize_strategy(s: Any) -> dict[str, Any]:
    """Serialize the strategy's public (factory-arg) fields.

    Emits ``{"type": ClsName, **factory_args}``; ``_``-prefixed fields and
    ``_COMPUTED_FIELDS`` (``sensitivity``) are skipped — the factory
    recomputes them on load.  Opaque torch tensors (e.g. ``lr_schedule``)
    are skipped too; the factory restores their default.
    """
    out: dict[str, Any] = {"type": s.__class__.__name__}
    for f in fields(s):
        if f.name.startswith("_") or f.name in _COMPUTED_FIELDS:
            continue
        v = getattr(s, f.name)
        if isinstance(v, _PRIMITIVE_TYPES):
            out[f.name] = v
        elif isinstance(v, (tuple, list)):
            out[f.name] = v
        # else: opaque (torch.Tensor for lr_schedule) — skip
    return out


def deserialize_strategy(sd: dict[str, Any]) -> Any:
    """Deserialize a strategy dict by calling its factory."""
    sd = dict(sd)
    t = sd.pop("type")
    cls = _STRATEGY_REGISTRY.get(t)
    if cls is None:
        raise ValueError(
            f"Unknown strategy type: {t!r} (registered: "
            f"{sorted(_STRATEGY_REGISTRY)!r})"
        )

    factory = _resolve_factory(t)
    return factory(**sd)


def register_strategy(cls: type) -> type:
    """Register a strategy dataclass for serialization.

    Indexes the class by name (so ``MfGaussian``'s custom deserializer
    can resolve a serialized strategy back to its class) and installs a
    ``state_dict`` / ``from_state_dict`` pair on the universal
    :mod:`opaque.serialization` registry so direct serialization of a
    bare strategy also works.

    Use as a decorator on each strategy dataclass.
    """
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
