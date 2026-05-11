"""Internal nested codec for :class:`DpProcess` (used by :mod:`opaque.serialization`).

Each process serializes as a nested dict carrying its concrete-class name in
the ``type`` field; inner :class:`DpProcess` fields recurse as sub-dicts.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from opaque.api.accounting.core._base import DpProcess


def _serialize_dp_process(p: DpProcess) -> dict[str, Any]:
    out: dict[str, Any] = {"type": p.__class__.__name__}
    for f in fields(p):
        v = getattr(p, f.name)
        if isinstance(v, DpProcess):
            out[f.name] = _serialize_dp_process(v)
        else:
            out[f.name] = v
    return out


def _load_dp_process(sd: dict[str, Any]) -> DpProcess:
    from opaque.api.accounting.core._base import _PROCESS_REGISTRY

    sd = dict(sd)
    t = sd.pop("type")
    cls = _PROCESS_REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown process type: {t}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        try:
            raw = sd.pop(f.name)
        except KeyError as e:
            raise ValueError(
                f"missing field {f.name!r} for {cls.__name__} "
                f"(remaining keys: {sorted(sd)!r})"
            ) from e
        if isinstance(raw, dict) and "type" in raw:
            kwargs[f.name] = _load_dp_process(raw)
        else:
            kwargs[f.name] = _coerce_field(cls, f.name, raw)

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
