"""Opaque: Functional DP-SGD for PyTorch.

Composable primitives for differentially private model training: per-example
gradient clipping, calibrated noise injection, privacy accounting, and Poisson
sampling. Built on ``torch.func`` with explicit state.

Modules:

- ``opaque.clipping``: Per-example gradient clipping (clipped_grad, clipped_fun, clip_pytree)
- ``opaque.noise``: Gaussian noise, bounded Gaussian, matrix-factorization correlated noise
- ``opaque.accounting``: PLD-based privacy accounting with composition, calibration, and metrics
- ``opaque.sampling``: Poisson, truncated Poisson, and cyclic Poisson samplers
- ``opaque.auditing``: Empirical privacy auditing via membership inference
- ``opaque.distributed``: DDP utilities (gradient aggregation, state sync)
- ``opaque.compat``: HuggingFace auto-patching for vmap compatibility
"""

import os

from opaque import accounting, auditing, distributed, sampling
from opaque.clipping import (
    AdaptiveClipState,
    ClipState,
    FixedClipState,
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
    CyclicPoissonSampler,
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
    # Sampling
    "sampling",
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "CyclicPoissonSampler",
    # Noise
    "gaussian_noise",
    "bounded_gaussian_noise",
    "band_mf_noise",
    "blt_mf_noise",
    "custom_mf_noise",
    "dense_mf_noise",
    "identity_mf_noise",
    # Accounting
    "accounting",
    # Auditing
    "auditing",
    # Distributed
    "distributed",
    # Utils
    "make_functional",
]
