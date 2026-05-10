"""Generic Python container walker (no third-party deps).

Handles the language-level container types — dataclasses, NamedTuples,
tuples, lists, dicts, primitives — that are universally serializable
without knowing anything about the leaf type. Specific concrete leaf
types (``torch.Tensor``, ``numpy.ndarray``, custom state objects) live
in the registry instead, so the dispatcher consults the registry first
and falls back here for anything generic.

Keeping this module torch-free is what allows ``opaque-base`` to ship
without a torch dependency. The torch and NumPy leaf handlers are
registered by ``opaque-engine`` and pure ndarray handlers are also
needed by ``opaque-accounting``; both register against the registry
on import.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Callable

WalkSave = Callable[[Any, str, dict[str, Any]], None]
WalkLoad = Callable[[Any, Mapping[str, Any], str], Any]

# Python primitives we serialise verbatim. Anything that is not a
# registered exact-type AND not one of these AND not a generic container
# is treated as opaque and skipped on save (the template-driven load
# preserves it).
_PRIMITIVES = (int, float, bool, str, type(None))


def _is_named_tuple_instance(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def walk_save(state: Any, prefix: str, out: dict[str, Any], recurse: WalkSave) -> None:
    """Save generic Python containers; opaque non-container leaves skip."""
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        for f in dataclasses.fields(state):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            recurse(getattr(state, f.name), sub, out)
        return
    if _is_named_tuple_instance(state):
        for f in state._fields:
            sub = f"{prefix}.{f}" if prefix else f
            recurse(getattr(state, f), sub, out)
        return
    if isinstance(state, tuple):
        for i, v in enumerate(state):
            recurse(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, list):
        for i, v in enumerate(state):
            recurse(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, dict):
        for k, v in state.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            recurse(v, sub, out)
        return
    if isinstance(state, _PRIMITIVES):
        out[prefix] = state
        return
    # Opaque — skip.


def walk_load(
    template: Any,
    sd: Mapping[str, Any],
    prefix: str,
    recurse: WalkLoad,
) -> Any:
    """Rebuild generic Python containers from ``sd``; pass opaque leaves through."""
    if dataclasses.is_dataclass(template) and not isinstance(template, type):
        replacements: dict[str, Any] = {}
        for f in dataclasses.fields(template):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            replacements[f.name] = recurse(getattr(template, f.name), sd, sub)
        return dataclasses.replace(template, **replacements)
    if _is_named_tuple_instance(template):
        return type(template)(
            *(
                recurse(getattr(template, f), sd, f"{prefix}.{f}" if prefix else f)
                for f in template._fields
            )
        )
    if isinstance(template, tuple):
        return tuple(recurse(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template))
    if isinstance(template, list):
        return [recurse(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template)]
    if isinstance(template, dict):
        return {
            k: recurse(v, sd, f"{prefix}.{k}" if prefix else str(k))
            for k, v in template.items()
        }
    if isinstance(template, _PRIMITIVES):
        return sd.get(prefix, template)
    # Opaque — pass through.
    return template


__all__ = ["walk_save", "walk_load", "WalkSave", "WalkLoad"]
