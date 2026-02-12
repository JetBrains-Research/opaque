"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy.

Note: Privacy accounting is provided by jbr-fed-accounting (external library).
"""

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
