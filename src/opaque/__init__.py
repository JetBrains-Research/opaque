"""Opaque: Differentially Private Training for PyTorch.

This package provides differentially private training utilities for PyTorch,
inspired by JAX-Privacy.
"""

from opaque.accounting import (
    PLDAccountant,
    RDPAccountant,
    calibrate_batch_size,
    calibrate_noise_multiplier,
    calibrate_steps,
)
from opaque.clipping import (
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.noise import add_gaussian_noise
from opaque.utils import make_functional

__all__ = [
    # Clipping
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    # Accounting
    "PLDAccountant",
    "RDPAccountant",
    "calibrate_noise_multiplier",
    "calibrate_steps",
    "calibrate_batch_size",
    # Noise
    "add_gaussian_noise",
    # Utils
    "make_functional",
]
