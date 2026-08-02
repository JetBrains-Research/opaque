"""Tensor and ndarray handler registration with the base serialization registry.

Imported on engine load via ``opaque.api.engine.__init__`` for its
import-time side effect: registers ``torch.Tensor``,
``torch.nn.Parameter``, ``numpy.ndarray``, and ``optree.PyTreeSpec``
with the ``opaque.api.base.serialization`` registry. The base wheel
itself ships only the generic Python container walker (dataclass,
NamedTuple, tuple, list, dict, primitives) and stays torch-free; this
module is where the torch substrate plugs torch leaves into the
unified serialization surface.

``nn.Parameter`` would already resolve to the ``torch.Tensor`` handler
through the dispatcher's MRO walk, but that handler returns a plain
tensor.  Parameters are what ``make_functional`` hands back, so they
get their own load handler that keeps the subclass and the
``requires_grad`` flag the template carries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optree
import torch

from opaque.api.base.serialization import (
    register_serializer,
    register_template_restored,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _tensor_save(obj: torch.Tensor) -> dict[str, Any]:
    return {"": obj.detach().clone()}


def _tensor_load(template: torch.Tensor, sd: Mapping[str, Any]) -> torch.Tensor:
    saved = sd.get("")
    if saved is None:
        return template
    if not isinstance(saved, torch.Tensor):
        raise TypeError(
            f"state_dict value expected a torch.Tensor, got {type(saved).__name__}"
        )
    return saved.to(dtype=template.dtype, device=template.device)


def _parameter_load(
    template: torch.nn.Parameter,
    sd: Mapping[str, Any],
) -> torch.nn.Parameter:
    restored = _tensor_load(template, sd)
    if restored is template:
        return template
    return torch.nn.Parameter(
        restored.detach(),
        requires_grad=template.requires_grad,
    )


def _ndarray_save(obj: np.ndarray) -> dict[str, Any]:
    return {"": obj.copy()}


def _ndarray_load(template: np.ndarray, sd: Mapping[str, Any]) -> np.ndarray:
    saved = sd.get("")
    if saved is None:
        return template
    arr = np.asarray(saved)
    if arr.shape != template.shape:
        raise ValueError(
            f"state_dict ndarray has shape {arr.shape}; "
            f"template expects {template.shape}"
        )
    if arr.dtype != template.dtype:
        arr = arr.astype(template.dtype, copy=False)
    return arr.copy()


register_serializer(torch.Tensor, _tensor_save, _tensor_load)
register_serializer(torch.nn.Parameter, _tensor_save, _parameter_load)
register_serializer(np.ndarray, _ndarray_save, _ndarray_load)

# A PyTreeSpec is the shape of a tree, not a value carried across steps:
# AdafactorState.treespec and the MF streaming-matrix states rebuild it
# from the parameter tree they are initialised against.
register_template_restored(optree.PyTreeSpec)


__all__: list[str] = []
