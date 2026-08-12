"""Neutral ndarray and structure handler registration with the base registry.

Imported on engine load via ``opaque.api.engine.__init__`` for its
import-time side effect: registers ``numpy.ndarray`` and ``optree.PyTreeSpec``
with the ``opaque.api.base.serialization`` registry. The base wheel
itself ships only the generic Python container walker (dataclass,
NamedTuple, tuple, list, dict, primitives) and stays framework-free. Native
array handlers live with their Torch, JAX, or MLX provider and register when
that provider loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optree

from opaque.api.base.serialization import (
    register_serializer,
    register_template_restored,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


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


register_serializer(np.ndarray, _ndarray_save, _ndarray_load)

# A PyTreeSpec is the shape of a tree, not a value carried across steps:
# AdafactorState.treespec and the MF streaming-matrix states rebuild it
# from the parameter tree they are initialised against.
register_template_restored(optree.PyTreeSpec)


__all__: list[str] = []
