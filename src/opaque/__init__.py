"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy.
"""

from opaque import accounting
from opaque.clipping import (
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.noise import add_gaussian_noise
from opaque.optimizers import (
    AdaptiveClipState,
    DPOptimizerState,
    dp_adam,
    dp_adam_ac,
    dp_adamw,
    dp_sgd,
    make_dp_optimizer,
)
from opaque.utils import make_functional

__all__ = [
    # Clipping
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    # Accounting (use opaque.accounting.* for functional API)
    "accounting",
    # Noise
    "add_gaussian_noise",
    # Optimizers
    "DPOptimizerState",
    "AdaptiveClipState",
    "make_dp_optimizer",
    "dp_sgd",
    "dp_adam",
    "dp_adamw",
    "dp_adam_ac",
    # Utils
    "make_functional",
]
