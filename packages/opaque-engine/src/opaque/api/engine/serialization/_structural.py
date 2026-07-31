"""Tensor and ndarray handler registration with the base serialization registry.

Imported on engine load via ``opaque.api.engine.__init__`` for its
import-time side effect: registers ``torch.Tensor`` and
``numpy.ndarray`` handlers, plus ``optree.PyTreeSpec`` as an inert
(template-restored) leaf, with the ``opaque.api.base.serialization``
registry. The base wheel itself ships only the generic Python container
walker (dataclass, NamedTuple, tuple, list, dict, primitives) and stays
torch-free; this module is where the torch substrate plugs torch leaves
into the unified serialization surface.

The registry resolves along ``__mro__``, so the ``torch.Tensor`` handler
also covers tensor subclasses — ``torch.nn.Parameter``, as returned by
:func:`opaque.functional.make_functional`, most of all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optree
import torch

from opaque.api.base.serialization import register_inert_type, register_serializer

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
    if saved.shape != template.shape:
        raise ValueError(
            f"state_dict tensor has shape {tuple(saved.shape)}; "
            f"template expects {tuple(template.shape)}"
        )
    loaded = saved.to(dtype=template.dtype, device=template.device)
    if isinstance(template, torch.nn.Parameter):
        return torch.nn.Parameter(loaded, requires_grad=template.requires_grad)
    return loaded


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
register_serializer(np.ndarray, _ndarray_save, _ndarray_load)

# A tree spec describes layout, not values: the restore template carries
# the same structure, so there is nothing to write to the state dict.
register_inert_type(optree.PyTreeSpec)


__all__: list[str] = []
