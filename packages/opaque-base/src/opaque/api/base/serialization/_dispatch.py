"""Dispatcher: registry-first, generic-structural fallback.

For each visited node the dispatcher consults the registry by exact
type. If a serializer pair is registered, it is used; otherwise the
generic structural walker handles the node (containers, primitives) or
skips it (opaque non-containers).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import _structural
from ._registry import _REGISTRY


def _join_path(prefix: str, rel_key: str) -> str:
    """Path join matching the structural walker's conventions.

    Dot segments separate field names; bracket-segments (``[i]``) attach
    directly without a leading dot. An empty ``rel_key`` collapses to
    ``prefix`` itself (a registered handler may emit ``{"": value}`` to
    occupy the prefix slot — used by leaf handlers like ``torch.Tensor``).
    """
    if not prefix:
        return rel_key
    if not rel_key:
        return prefix
    if rel_key.startswith("["):
        return prefix + rel_key
    return f"{prefix}.{rel_key}"


def _subdict(sd: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Slice ``sd`` to keys nested under ``prefix`` (dot + bracket segments)."""
    if not prefix:
        return dict(sd)
    out: dict[str, Any] = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k == prefix:
            out[""] = v
        elif k.startswith(prefix) and len(k) > plen:
            sep = k[plen]
            if sep == ".":
                out[k[plen + 1 :]] = v
            elif sep == "[":
                out[k[plen:]] = v
    return out


def _walk_save(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    cls = type(obj)
    if cls in _REGISTRY:
        rel = _REGISTRY[cls][0](obj)
        for rk, rv in rel.items():
            out[_join_path(prefix, rk)] = rv
        return
    _structural.walk_save(obj, prefix, out, _walk_save)


def _walk_load(template: Any, sd: Mapping[str, Any], prefix: str) -> Any:
    cls = type(template)
    if cls in _REGISTRY:
        return _REGISTRY[cls][1](template, _subdict(sd, prefix))
    return _structural.walk_load(template, sd, prefix, _walk_load)


def state_dict(obj: Any) -> dict[str, Any]:
    """Serialise ``obj`` to a flat ``dict[str, Any]``.

    Registered types use their exact-type handler; everything else is
    walked structurally (dataclass, NamedTuple, tuple, list, dict,
    primitives). Opaque non-containers are dropped — the load is
    template-driven and reads them back from the template.
    """
    out: dict[str, Any] = {}
    _walk_save(obj, "", out)
    return out


def from_state_dict(template: Any, sd: Mapping[str, Any]) -> Any:
    """Rebuild from ``sd`` using ``template`` for shape and omitted leaves.

    For types registered with :func:`register_serializer`, the
    ``template`` argument selects the handler; handlers may ignore it
    when the dict is self-describing (e.g. ``DpProcess`` subclasses).
    Generic Python container shapes come from the template; missing
    leaves keep the template's value (forward compatibility when new
    fields appear).
    """
    return _walk_load(template, sd, "")


__all__ = ["state_dict", "from_state_dict"]
