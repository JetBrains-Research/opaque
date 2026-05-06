"""Structural tree (de)serialisation — tensors, dataclasses, named tuples, containers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Callable

import torch

WalkSave = Callable[[Any, str, dict[str, Any]], None]
WalkLoad = Callable[[Any, Mapping[str, Any], str], Any]

# Python primitives we serialise verbatim.  Everything else that isn't a
# tensor / dataclass / collection / named tuple is treated as opaque and
# skipped on save.
_PRIMITIVES = (int, float, bool, str, type(None))


def _is_named_tuple_instance(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def walk_save(state: Any, prefix: str, out: dict[str, Any], recurse: WalkSave) -> None:
    if isinstance(state, torch.Tensor):
        out[prefix] = state.detach().clone()
        return
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
    template: Any, sd: Mapping[str, Any], prefix: str, recurse: WalkLoad
) -> Any:
    if isinstance(template, torch.Tensor):
        saved = sd.get(prefix)
        if saved is None:
            return template
        if not isinstance(saved, torch.Tensor):
            raise TypeError(
                f"state_dict[{prefix!r}] expected a torch.Tensor, "
                f"got {type(saved).__name__}"
            )
        return saved.to(dtype=template.dtype, device=template.device)
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
