"""Internal nested codec for :class:`DpProcess` (used by :mod:`opaque.serialization`).

Each process serializes as a nested dict carrying its concrete-class name in
the ``type`` field; inner :class:`DpProcess` and dataclass fields recurse
as sub-dicts.  Opaque (non-primitive, non-dataclass, non-container) values
are skipped on save and reconstituted from the class default on load.

Strategy dataclasses (recipes attached to ``MfGaussian.strategy`` etc.)
register their class name → class mapping via :func:`register_strategy`
so the loader can resolve ``"type": "BltStrategy"`` back to the concrete
class.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from typing import Any

from opaque.api.accounting.core._base import DpProcess  # noqa: F401  (used downstream)

#: Type-name → class for non-DpProcess dataclasses (strategies) that may
#: appear as a field on a registered :class:`DpProcess`.  Populated at
#: import time of each strategy module via :func:`register_strategy`.
_STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(cls: type) -> type:
    """Register a non-DpProcess dataclass for serialization.

    Returns ``cls`` for use as a decorator.
    """
    _STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


_PRIMITIVE_TYPES = (int, float, bool, str, type(None))


def _serialize_dp_process(p: Any) -> dict[str, Any]:
    """Serialize a DpProcess or registered strategy to a nested dict.

    Recurses into nested :class:`DpProcess` and registered strategy
    dataclasses.  Skips opaque values (e.g. ``StreamingMatrix``,
    ``torch.Tensor``, unregistered dataclasses) so the on-wire dict
    carries only primitives + nested known dataclasses.  Skipped fields
    restore to their class default on load.
    """
    from opaque.api.accounting.core._base import _PROCESS_REGISTRY

    out: dict[str, Any] = {"type": p.__class__.__name__}
    for f in fields(p):
        v = getattr(p, f.name)
        cls_name = type(v).__name__
        if isinstance(v, DpProcess):
            out[f.name] = _serialize_dp_process(v)
        elif cls_name in _STRATEGY_REGISTRY or cls_name in _PROCESS_REGISTRY:
            out[f.name] = _serialize_dp_process(v)
        elif isinstance(v, _PRIMITIVE_TYPES):
            out[f.name] = v
        elif isinstance(v, (tuple, list)):
            out[f.name] = v
        # else: opaque (e.g. StreamingMatrix, torch.Tensor) — skip
    return out


def _load_dp_process(sd: dict[str, Any]) -> Any:
    from opaque.api.accounting.core._base import _PROCESS_REGISTRY

    sd = dict(sd)
    t = sd.pop("type")
    cls = _PROCESS_REGISTRY.get(t) or _STRATEGY_REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown process or strategy type: {t}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in sd:
            raw = sd.pop(f.name)
            if isinstance(raw, dict) and "type" in raw:
                kwargs[f.name] = _load_dp_process(raw)
            else:
                kwargs[f.name] = _coerce_field(cls, f.name, raw)
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()
        else:
            raise ValueError(f"missing required field {f.name!r} for {cls.__name__}")

    if sd:
        raise ValueError(f"unexpected keys for {cls.__name__}: {sorted(sd)!r}")

    return cls(**kwargs)


def _coerce_field(cls: type[Any], fname: str, val: Any) -> Any:
    from typing import get_origin, get_type_hints

    try:
        hints = get_type_hints(cls)
    except Exception:  # noqa: BLE001
        return val
    t = hints.get(fname)
    if t is None:
        return val
    origin = get_origin(t)
    if origin is tuple and isinstance(val, list):
        return tuple(val)
    return val


__all__: list[str] = []
