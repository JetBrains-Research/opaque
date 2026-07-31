"""Dispatcher: registry-first, generic-structural fallback, fail closed.

For each visited node the dispatcher resolves the registry along the
node's ``__mro__`` (exact type first, then base classes). A resolved
serializer pair handles the node; a type declared inert is omitted on
save and taken from the template on load; otherwise the generic
structural walker handles containers and primitives.

A leaf that matches none of those raises :exc:`TypeError`. Skipping it
would drop state from the checkpoint without a trace, which for noise /
accountant / optimizer state means a restored run silently differs from
the one that was saved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _structural
from ._registry import is_inert, resolve_serializer

if TYPE_CHECKING:
    from collections.abc import Mapping


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


def _unhandled(obj: Any, prefix: str, *, op: str) -> TypeError:
    """Error for a leaf that resolves to no handler and is not a container."""
    where = f" at {prefix!r}" if prefix else ""
    return TypeError(
        f"{op} cannot handle {type(obj).__qualname__}{where}: no serializer is "
        "registered for it or any of its base classes, and it is not a generic "
        "container. Register a handler with "
        "opaque.serialization.register_serializer(), or declare it stateless "
        "with opaque.serialization.register_inert_type() when the template "
        "value is authoritative."
    )


def _walk_save(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    pair = resolve_serializer(type(obj))
    if pair is not None:
        rel = pair[0](obj)
        for rk, rv in rel.items():
            out[_join_path(prefix, rk)] = rv
        return
    if is_inert(type(obj)):
        return
    if not _structural.is_structural(obj):
        raise _unhandled(obj, prefix, op="state_dict()")
    _structural.walk_save(obj, prefix, out, _walk_save)


def _walk_load(template: Any, sd: Mapping[str, Any], prefix: str) -> Any:
    pair = resolve_serializer(type(template))
    if pair is not None:
        return pair[1](template, _subdict(sd, prefix))
    if is_inert(type(template)):
        return template
    if not _structural.is_structural(template):
        raise _unhandled(template, prefix, op="from_state_dict()")
    return _structural.walk_load(template, sd, prefix, _walk_load)


def state_dict(obj: Any) -> dict[str, Any]:
    """Serialise ``obj`` to a flat ``dict[str, Any]``.

    Registered types (and their subclasses) use their registered
    handler; everything else is walked structurally (dataclass,
    NamedTuple, tuple, list, dict, primitives). Types declared with
    :func:`~opaque.serialization.register_inert_type` are omitted and
    read back from the template.

    Raises:
        TypeError: A leaf resolves to no handler and is not a generic
            container. Silently dropping it would leave a checkpoint
            that restores different state than it saved.
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

    Raises:
        TypeError: Mirrors :func:`state_dict` — a template leaf that
            resolves to no handler and is not a generic container.
    """
    return _walk_load(template, sd, "")


__all__ = ["from_state_dict", "state_dict"]
