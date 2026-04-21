"""Opaque DP-SGD: Differentially Private SGD mechanisms.

Gaussian / truncated-Gaussian noise, per-group noise allocation, adaptive
and AUTO-S clipping, the standard + truncated Poisson samplers, and the
AdamW-BC optimizer. Fixed-clipping primitives used by this package live in
:mod:`opaque.clipping`.

Data classes (``AdaptiveClipState``, ``AdaptiveClippedGradAux``,
``AutoClipState``, ``AutoClippedGradAux``) are importable from this module
for type annotations but are not part of ``__all__`` — the public surface
is functional.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.dpsgd import clipping, noise, optimizers, sampling
from opaque.dpsgd.clipping.adaptive import (
    AdaptiveClippedGradAux as AdaptiveClippedGradAux,
)
from opaque.dpsgd.clipping.adaptive import AdaptiveClipState as AdaptiveClipState
from opaque.dpsgd.clipping.adaptive import adaptive_clipped_grad
from opaque.dpsgd.clipping.auto import AutoClippedGradAux as AutoClippedGradAux
from opaque.dpsgd.clipping.auto import AutoClipState as AutoClipState
from opaque.dpsgd.clipping.auto import auto_clipped_grad

# Side-effect import: registers DP-SGD-specific sync handlers (AdaptiveClipState /
# AdaptiveClippedGradAux) with opaque.distributed.sync(). Reach the helpers
# directly via opaque.dpsgd.clipping.distributed if needed.
from opaque.dpsgd.clipping import distributed as _clipping_distributed  # noqa: F401
from opaque.dpsgd.noise.gaussian import gaussian_noise
from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev
from opaque.dpsgd.noise.truncated_gaussian import truncated_gaussian_noise
from opaque.dpsgd.optimizers.adamw_bc import adamw_bc
from opaque.dpsgd.sampling.poisson import PoissonSampler
from opaque.dpsgd.sampling.truncated_poisson import TruncatedPoissonSampler

try:
    __version__ = _pkg_version("opaque-dpsgd")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Subpackages
    "clipping",
    "noise",
    "optimizers",
    "sampling",
    # Clipping (DP-SGD-specific; fixed-clipping at opaque.clipping)
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    # Noise mechanisms
    "gaussian_noise",
    "truncated_gaussian_noise",
    "per_group_noise_stddev",
    # Sampling
    "PoissonSampler",
    "TruncatedPoissonSampler",
    # Optimizer
    "adamw_bc",
]
