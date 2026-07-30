"""Internal nested codec for :class:`DpProcess` (used by :mod:`opaque.serialization`).

Each process serializes as a nested dict carrying its concrete-class name in
the ``type`` field; inner :class:`DpProcess` fields recurse via the same
codec.  Non-``DpProcess`` field values are emitted only for primitives and
plain containers (tuple, list) — opaque values (e.g. ``torch.Tensor``)
are skipped on save and restored to the class default on load.

The codec knows only about :class:`DpProcess`.  Mechanisms that wrap
domain-specific sub-objects (strategies, encoders, …) register their own
custom ``(state_dict_fn, from_state_dict_fn)`` with
:func:`opaque.serialization.register_serializer`; that registration
replaces the generic codec for that class.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from typing import Any

from opaque.api.accounting.core._base import DpProcess

_PRIMITIVE_TYPES = (int, float, bool, str, type(None))


def _serialize_dp_process(p: Any) -> dict[str, Any]:
    """Serialize a DpProcess (or nested DpProcess sub-object) to a nested dict.

    Each field is dispatched by value:

    - Nested ``DpProcess`` → dispatch via :mod:`opaque.serialization` so
      custom serializers (e.g. ``MfGaussian``'s strategy-aware one) fire.
    - Primitive / tuple / list → emit verbatim.
    - Anything else → skip (load restores the class default).
    """
    from opaque.serialization import state_dict as opaque_state_dict

    out: dict[str, Any] = {"type": p.__class__.__name__}
    for f in fields(p):
        v = getattr(p, f.name)
        if isinstance(v, DpProcess):
            out[f.name] = dict(opaque_state_dict(v))
        elif isinstance(v, (_PRIMITIVE_TYPES, tuple, list)):
            out[f.name] = v
        # else: opaque — skip
    return out


def _load_dp_process(sd: dict[str, Any]) -> Any:
    """Deserialize a nested ``{"type": ..., ...}`` dict produced by
    :func:`_serialize_dp_process`.

    Nested ``{"type": X, ...}`` sub-dicts dispatch through the universal
    :mod:`opaque.serialization` registry so custom serializers fire.
    """
    from opaque.api.accounting.core._base import _PROCESS_REGISTRY
    from opaque.serialization import from_state_dict as opaque_from_state_dict

    sd = dict(sd)
    t = sd.pop("type")
    cls = _PROCESS_REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown DpProcess type: {t}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in sd:
            raw = sd.pop(f.name)
            if isinstance(raw, dict) and "type" in raw:
                inner_cls = _PROCESS_REGISTRY.get(raw["type"])
                if inner_cls is None:
                    raise ValueError(
                        f"Unknown nested DpProcess type {raw['type']!r} "
                        f"in field {f.name!r} of {cls.__name__}"
                    )
                template = inner_cls.__new__(inner_cls)
                kwargs[f.name] = opaque_from_state_dict(template, raw)
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
    except Exception:
        return val
    t = hints.get(fname)
    if t is None:
        return val
    origin = get_origin(t)
    if origin is tuple and isinstance(val, list):
        return tuple(val)
    return val


__all__: list[str] = []
