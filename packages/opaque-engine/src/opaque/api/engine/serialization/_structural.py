"""Neutral ndarray and structure handler registration with the base registry.

Imported on engine load via ``opaque.api.engine.__init__`` for its
import-time side effect: registers ``numpy.ndarray`` and ``optree.PyTreeSpec``
with the ``opaque.api.base.serialization`` registry. The base wheel
itself ships only the generic Python container walker (dataclass,
NamedTuple, tuple, list, dict, primitives) and stays framework-free. Native
array handlers live with their Torch, JAX, or MLX provider and register when
that provider loads.

Restore is template-driven and each attribute of an array leaf follows
its own rule. Shape must match the template exactly — a
broadcast-compatible mismatch would otherwise restore silently — and each
provider applies the same rule to its own array leaves. Dtype is taken
from the template, so a checkpoint written in one compute dtype resumes
in the live one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optree

from opaque.api.base.serialization import (
    register_fallback_resolver,
    register_serializer,
    register_template_restored,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _check_shape(saved: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """Reject a checkpoint value whose shape differs from the template's.

    The message normalises to ``tuple`` so a ``torch.Size`` and an ndarray
    shape read identically.
    """
    if saved != expected:
        raise ValueError(
            f"state_dict value has shape {tuple(saved)}; template expects "
            f"{tuple(expected)}. Restore is template-driven: rebuild the "
            "template from the configuration the checkpoint was written with."
        )


def _ndarray_save(obj: np.ndarray) -> dict[str, Any]:
    return {"": obj.copy()}


def _ndarray_load(template: np.ndarray, sd: Mapping[str, Any]) -> np.ndarray:
    saved = sd.get("")
    if saved is None:
        return template
    arr = np.asarray(saved)
    _check_shape(arr.shape, template.shape)
    if arr.dtype != template.dtype:
        arr = arr.astype(template.dtype, copy=False)
    return arr.copy()


register_serializer(np.ndarray, _ndarray_save, _ndarray_load)

# A PyTreeSpec is the shape of a tree, not a value carried across steps:
# AdafactorState.treespec and the MF streaming-matrix states rebuild it
# from the parameter tree they are initialised against.
register_template_restored(optree.PyTreeSpec)


def _activate_backend_for_unknown_leaf(value: Any) -> bool:
    """Serialization fallback: infer + activate a provider for ``value``.

    Provider activation registers native-array serializers, so a
    ``state_dict``/``from_state_dict`` call that meets a framework tensor
    before any explicit backend activation resolves instead of raising.
    """
    from opaque.api.engine.backend import BackendError, ensure_backend

    try:
        ensure_backend(value)
    except BackendError:
        return False
    return True


register_fallback_resolver(_activate_backend_for_unknown_leaf)


__all__: list[str] = []
