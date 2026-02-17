"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy. Includes standard DP-SGD as well as correlated noise
mechanisms (BandMF, BLT, DP-FTRL) for improved utility.

Note: Privacy accounting is provided by jbr-fed-accounting (external library).
"""

import os

from opaque import auditing, distributed, sampling
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
from opaque.noise import (
    band_mf_noise,
    blt_mf_noise,
    bounded_gaussian_noise,
    custom_mf_noise,
    dense_mf_noise,
    gaussian_noise,
    identity_mf_noise,
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
    "gaussian_noise",
    "bounded_gaussian_noise",
    "band_mf_noise",
    "blt_mf_noise",
    "custom_mf_noise",
    "dense_mf_noise",
    "identity_mf_noise",
    # Auditing
    "auditing",
    # Distributed
    "distributed",
    # Utils
    "make_functional",
]
