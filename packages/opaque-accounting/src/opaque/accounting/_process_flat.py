"""Flat :class:`DpProcess` (de)serialization for :mod:`opaque.serialization`.

Keys use dotted prefixes (``left.type``, ``inner.noise_multiplier``, …)
so accountant and process trees share the same flat ``dict[str, Any]``
convention as other Opaque runtime state.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from opaque.accounting._base import DpProcess


def flat_dp_process_state(p: DpProcess, prefix: str = "") -> dict[str, Any]:
    """Flatten *p* to string keys; each subtree root has a ``type`` entry."""
    out: dict[str, Any] = {f"{prefix}type": p.__class__.__name__}
    for f in fields(p):
        v = getattr(p, f.name)
        fp = f"{prefix}{f.name}"
        if isinstance(v, DpProcess):
            out.update(flat_dp_process_state(v, f"{fp}."))
        else:
            out[fp] = v
    return out


def load_dp_process(sd: dict[str, Any]) -> DpProcess:
    """Reconstruct a process from a flat dict (relative keys, rooted at ``type``)."""
    from opaque.accounting._base import _PROCESS_REGISTRY

    sd = dict(sd)
    t = sd.pop("type")
    cls = _PROCESS_REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown process type: {t}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        fp = f"{f.name}."
        if f"{f.name}.type" in sd:
            sub = {k[len(fp) :]: sd[k] for k in list(sd.keys()) if k.startswith(fp)}
            for k in list(sd.keys()):
                if k.startswith(fp):
                    del sd[k]
            kwargs[f.name] = load_dp_process(sub)
        else:
            try:
                raw = sd.pop(f.name)
            except KeyError as e:
                raise ValueError(
                    f"missing field {f.name!r} for {cls.__name__} "
                    f"(remaining keys: {sorted(sd)!r})"
                ) from e
            kwargs[f.name] = _coerce_field(cls, f.name, raw)

    if sd:
        raise ValueError(
            f"unexpected keys for {cls.__name__}: {sorted(sd)!r}"
        )

    return cls(**kwargs)


def dp_process_state_dict(p: DpProcess) -> dict[str, Any]:
    """Serialize *p* via :mod:`opaque.serialization` (flat keys)."""
    from opaque.serialization import state_dict as s

    return dict(s(p))


dp_process_from_state_dict = load_dp_process


def _coerce_field(cls: type[Any], fname: str, val: Any) -> Any:
    """Coerce JSON-adjacent values (e.g. list → tuple) using field annotations."""
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
