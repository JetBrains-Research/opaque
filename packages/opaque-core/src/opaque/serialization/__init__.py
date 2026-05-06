"""Functional serialization for Opaque explicit state objects.

Flattens tensor trees, NumPy arrays, dataclasses, named tuples, sequences, and
string-keyed dicts into a flat ``dict[str, Any]`` suitable for
``torch.save`` / ``torch.load``.  Non-serialisable leaves (vendor specs,
callables, …) are omitted on save and preserved from the *template*
passed to :func:`from_state_dict`.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf.  Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serialization_type`; the default is a structural
walk for every other type.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from opaque.serialization import _structural

#: ``state_dict_fn`` returns path keys *relative* to the object root (no
#: leading dot).  ``from_state_dict_fn`` receives the same relative-key dict.
_StateDictFn = Callable[[Any], dict[str, Any]]
_FromStateDictFn = Callable[[Any, dict[str, Any]], Any]

_REGISTRY: dict[type[Any], tuple[_StateDictFn, _FromStateDictFn]] = {}


def register_serialization_type(
    typ: type[Any],
    state_dict_fn: _StateDictFn,
    from_state_dict_fn: _FromStateDictFn,
) -> None:
    """Register custom (de)serialisation for ``typ`` (also used at nested nodes)."""
    _REGISTRY[typ] = (state_dict_fn, from_state_dict_fn)


def _subdict(sd: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return dict(sd)
    out: dict[str, Any] = {}
    sep = prefix + "."
    for k, v in sd.items():
        if k == prefix:
            out[""] = v
        elif k.startswith(sep):
            out[k[len(sep) :]] = v
    return out


def _walk_save(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    cls = type(obj)
    if cls in _REGISTRY:
        rel = _REGISTRY[cls][0](obj)
        for rk, rv in rel.items():
            full = f"{prefix}.{rk}" if prefix else rk
            out[full] = rv
        return
    _structural.walk_save(obj, prefix, out, _walk_save)


def _walk_load(template: Any, sd: Mapping[str, Any], prefix: str) -> Any:
    cls = type(template)
    if cls in _REGISTRY:
        return _REGISTRY[cls][1](template, _subdict(sd, prefix))
    return _structural.walk_load(template, sd, prefix, _walk_load)


def state_dict(obj: Any) -> dict[str, Any]:
    """Serialise ``obj`` to a flat ``dict[str, Any]``."""
    out: dict[str, Any] = {}
    _walk_save(obj, "", out)
    return out


def from_state_dict(template: Any, sd: Mapping[str, Any]) -> Any:
    """Rebuild from ``sd`` using ``template`` for shape and omitted leaves.

    For types registered with :func:`register_serialization_type`, the
    ``template`` argument selects the handler; handlers may ignore it when
    the dict is self-describing (as with :class:`~opaque.accounting._base.DpProcess`).
    """
    return _walk_load(template, sd, "")


__all__: Final[list[str]] = [
    "from_state_dict",
    "register_serialization_type",
    "state_dict",
]
