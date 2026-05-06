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

The :mod:`opaque.dpsgd.accounting` subpackage (DP-SGD-specific privacy
accounting factories, requires ``opaque-accounting``) is **lazy-imported**:
``import opaque.dpsgd; opaque.dpsgd.accounting.gaussian(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute access
— so callers that only need clipping / noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

from opaque.dpsgd import clipping, noise, sampling
from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad
from opaque.dpsgd.noise import (
    gaussian_noise,
    per_group_noise_stddev,
    truncated_gaussian_noise,
)
from opaque.dpsgd.sampling import PoissonSampler, TruncatedPoissonSampler

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opaque.dpsgd import accounting as accounting

try:
    __version__ = _pkg_version("opaque-dpsgd")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opaque.dpsgd.accounting``.

    Defers loading ``opaque.accounting`` (and its native Rust extension)
    until ``opaque.dpsgd.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opaque.dpsgd.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opaque.dpsgd' has no attribute {name!r}")


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
