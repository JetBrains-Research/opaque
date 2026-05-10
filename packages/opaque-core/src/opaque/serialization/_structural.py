"""Tensor and ndarray handler registration with the base serialization registry.

This module exists for its import-time side effect: it registers
``torch.Tensor`` and ``numpy.ndarray`` as exact-type handlers with the
``opaque.api.base.serialization`` registry. The base wheel itself
ships only the generic Python container walker (dataclass, NamedTuple,
tuple, list, dict, primitives) and stays torch-free.

In phase 2 of the refactor, the torch handler moves to
``opaque.api.engine.serialization`` and the numpy handler can be
folded in there too. This intermediate module disappears at that
point.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from opaque.api.base.serialization import register_serializer


def _tensor_save(obj: torch.Tensor) -> dict[str, Any]:
    return {"": obj.detach().clone()}


def _tensor_load(template: torch.Tensor, sd: Mapping[str, Any]) -> torch.Tensor:
    saved = sd.get("")
    if saved is None:
        return template
    if not isinstance(saved, torch.Tensor):
        raise TypeError(
            f"state_dict value expected a torch.Tensor, "
            f"got {type(saved).__name__}"
        )
    return saved.to(dtype=template.dtype, device=template.device)


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


__all__: list[str] = []
