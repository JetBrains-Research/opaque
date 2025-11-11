"""Opaque package bootstrap.

This project provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy. See `.junie/guidelines.md` for contributor guidance.
"""

from opaque.clipping import (
    AuxiliaryOutput,
    BoundedSensitivityCallable,
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.pytree_utils import global_norm, tree_leaves, tree_map

__all__ = [
    "AuxiliaryOutput",
    "BoundedSensitivityCallable",
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "global_norm",
    "tree_leaves",
    "tree_map",
]
