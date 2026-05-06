"""Opaque DP-SGD: Differentially Private SGD mechanisms.

Gaussian / truncated-Gaussian noise, per-group noise allocation, adaptive
and AUTO-S clipping, and the standard + truncated Poisson samplers.
Fixed-clipping primitives used by this package live in
:mod:`opaque.clipping`; functional optimizers (including the universal
``adamw`` with DP bias-correction and private second-moment paths) live in
:mod:`opaque.optimizers`.

State / aux dataclasses (``AdaptiveClipState``, ``AdaptiveClippedGradAux``,
``AutoClipState``, ``AutoClippedFunAux``, ``AutoClippedGradAux``) live in
:mod:`opaque.dpsgd.clipping.types`.  ``GaussianNoiseState`` lives in
:mod:`opaque.dpsgd.noise.types`.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.dpsgd import accounting, clipping, noise, sampling
from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad
from opaque.dpsgd.noise import (
    gaussian_noise,
    per_group_noise_stddev,
    truncated_gaussian_noise,
)
from opaque.dpsgd.sampling import PoissonSampler, TruncatedPoissonSampler

try:
    __version__ = _pkg_version("opaque-dpsgd")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Subpackages
    "accounting",
    "clipping",
    "noise",
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
]
