"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy. Includes standard DP-SGD as well as correlated noise
mechanisms (BandMF, BLT, DP-FTRL) for improved utility.

Note: Privacy accounting is provided by jbr-fed-accounting (external library).
"""

import os

from opaque import matrix_factorization, sampling
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
from opaque.dp_ftrl import DPFTRLOptimizer, DPFTRLState, dp_ftrl_train_step
from opaque.noise import (
    bounded_gaussian,
    bounded_gaussian_stateful,
    gaussian,
    gaussian_stateful,
)
from opaque.noise.matrix_factorization import (
    Privatizer,
    PrivatizerState,
    gaussian_privatizer,
    matrix_factorization_privatizer,
)
from opaque.sampling import (
    CyclicPoissonSampling,
    PoissonSampler,
    TruncatedPoissonSampler,
)
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
    "CyclicPoissonSampling",
    # Noise
    "bounded_gaussian",
    "bounded_gaussian_stateful",
    "gaussian",
    "gaussian_stateful",
    # Matrix Factorization / DP-FTRL
    "matrix_factorization",
    "matrix_factorization_privatizer",
    "gaussian_privatizer",
    "Privatizer",
    "PrivatizerState",
    "DPFTRLOptimizer",
    "DPFTRLState",
    "dp_ftrl_train_step",
    # Utils
    "make_functional",
]
