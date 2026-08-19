"""Torch exact-type serialization handlers for the engine provider.

Restore is template-driven and each attribute of an array leaf follows
its own rule. Shape must match the template exactly — a
broadcast-compatible mismatch would otherwise restore silently — which is
the same rule the engine applies to its ``ndarray`` leaves. Dtype and
device are taken from the template, so a checkpoint read onto the CPU
resumes in the live compute dtype on the training device.
"""

from __future__ import annotations

from typing import Any

import torch
from opaque.api.base.serialization import register_serializer


def _check_shape(saved: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """Reject a checkpoint value whose shape differs from the template's.

    The message normalises to ``tuple`` so a ``torch.Size`` and an ndarray
    shape read identically.
    """
    if tuple(saved) != tuple(expected):
        raise ValueError(
            f"state_dict value has shape {tuple(saved)}; template expects "
            f"{tuple(expected)}. Restore is template-driven: rebuild the "
            "template from the configuration the checkpoint was written with."
        )


def _tensor_save(obj: torch.Tensor) -> dict[str, Any]:
    return {"": obj.detach().clone()}


def _tensor_load(template: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    saved = state.get("")
    if saved is None:
        return template
    if not isinstance(saved, torch.Tensor):
        raise TypeError(
            f"state_dict value expected a torch.Tensor, got {type(saved).__name__}"
        )
    _check_shape(saved.shape, template.shape)
    return saved.to(dtype=template.dtype, device=template.device)


def _parameter_load(
    template: torch.nn.Parameter, state: dict[str, Any]
) -> torch.nn.Parameter:
    restored = _tensor_load(template, state)
    if restored is template:
        return template
    return torch.nn.Parameter(restored.detach(), requires_grad=template.requires_grad)


def register_torch_serialization() -> None:
    """Register Torch tensor and parameter handlers with the base registry."""
    register_serializer(torch.Tensor, _tensor_save, _tensor_load)
    register_serializer(torch.nn.Parameter, _tensor_save, _parameter_load)


__all__ = ["register_torch_serialization"]
