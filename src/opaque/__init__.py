"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy.

Note: Privacy accounting is provided by jbr-fed-accounting (external library).
"""

import os

from opaque import sampling
from opaque.clipping import (
    AdaptiveClipState,
    ClipState,
    FixedClipState,
    NeighboringRelation,
    adaptive_clipped_grad,
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.noise import gaussian, gaussian_stateful
from opaque.sampling import PoissonSampler, TruncatedPoissonSampler

# TEMPORARILY COMMENTED OUT - optimizers being refactored to functional API
# from opaque.optimizers import (
#     AdaptiveClipState,
#     DPOptimizerState,
#     dp_adam,
#     dp_adam_ac,
#     dp_adamw,
#     dp_sgd,
#     make_dp_optimizer,
# )
from opaque.utils import make_functional

# =============================================================================
# Auto-patching for HuggingFace Transformers vmap compatibility
# =============================================================================
# Disable with: OPAQUE_NO_PATCH=1
#
# These patches make HuggingFace models work with vmap (required for clipped_grad).
# They replace functions that use hardcoded shapes or data-dependent control flow
# with vmap-compatible versions using dynamic shapes.

if not os.environ.get("OPAQUE_NO_PATCH"):
    try:
        from opaque.compat.transformers import apply_global_patches

        apply_global_patches()
    except ImportError:
        # transformers not installed, skip patching
        pass

__all__ = [
    # Clipping
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "adaptive_clipped_grad",
    "ClipState",
    "FixedClipState",
    "AdaptiveClipState",
    "NeighboringRelation",
    # Sampling
    "sampling",
    "PoissonSampler",
    "TruncatedPoissonSampler",
    # Noise
    "gaussian",
    "gaussian_stateful",
    # Utils
    "make_functional",
]
