"""Torch provider façade."""

from opaque.api.torch._transforms import under_functorch_transform
from opaque.api.torch.backend import torch_backend
from opaque.api.torch.patches import apply_runtime_patches

__all__ = [
    "apply_runtime_patches",
    "torch_backend",
    "under_functorch_transform",
]
