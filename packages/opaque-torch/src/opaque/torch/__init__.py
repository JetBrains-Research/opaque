"""Torch provider façade.

:func:`torch_backend` names the provider for
:func:`opaque.backend.set_backend`, and :func:`apply_runtime_patches`
installs the Torch-core runtime shims. The patch-author probes live in
the submodule façades — :mod:`opaque.torch.transforms` for the
``torch.func`` interpreter stack, :mod:`opaque.torch.checkpoint` for the
gradient-checkpointing installers.
"""

from opaque.api.torch.backend import torch_backend
from opaque.api.torch.patches import apply_runtime_patches

__all__ = [
    "apply_runtime_patches",
    "torch_backend",
]
