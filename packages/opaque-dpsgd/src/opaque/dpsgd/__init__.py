"""Opaque DP-SGD: Differentially Private SGD mechanisms.

Gaussian / truncated-Gaussian noise, adaptive and AUTO-S clipping,
truncated Poisson sampling, and the AdamW-BC optimizer. All depend on
primitives in :mod:`opaque.core`.
"""

from opaque.dpsgd import clipping, noise, optimizers, sampling
from opaque.dpsgd.clipping.adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.dpsgd.clipping.auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    auto_clipped_fun,
    auto_clipped_grad,
)
from opaque.dpsgd.clipping.distributed import (
    sync_adaptive_clip_state,
    sync_adaptive_clipped_grad_aux,
)
from opaque.dpsgd.noise.gaussian import gaussian_noise
from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev
from opaque.dpsgd.noise.truncated_gaussian import truncated_gaussian_noise
from opaque.dpsgd.optimizers.adamw_bc import adamw_bc
from opaque.dpsgd.sampling.truncated_poisson import TruncatedPoissonSampler

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # Subpackages
    "clipping",
    "noise",
    "optimizers",
    "sampling",
    # Clipping
    "adaptive_clipped_grad",
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "auto_clipped_fun",
    "auto_clipped_grad",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "sync_adaptive_clip_state",
    "sync_adaptive_clipped_grad_aux",
    # Noise mechanisms
    "gaussian_noise",
    "truncated_gaussian_noise",
    "per_group_noise_stddev",
    # Sampling
    "TruncatedPoissonSampler",
    # Optimizers
    "adamw_bc",
]
