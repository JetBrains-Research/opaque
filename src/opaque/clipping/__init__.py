"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping gradients and function outputs
to bounded L2 sensitivity, a key requirement for DP-SGD.
"""

from opaque.clipping.clipped_fun import clipped_fun
from opaque.clipping.clipped_grad import clipped_grad
from opaque.clipping.pytree import clip_pytree
from opaque.clipping.types import AuxiliaryOutput, BoundedSensitivityCallable

__all__ = [
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "AuxiliaryOutput",
    "BoundedSensitivityCallable",
]
