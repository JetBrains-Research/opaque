"""Opaque DP-FTRL: correlated (matrix-factorization) noise mechanisms.

Strategies (BLT, BSR, BiSR, band-MF, λ-CGD, identity) + DP-FTRL-specific
participation samplers (b-min-sep, cyclic Poisson, balls-in-bins, sequential).

Fixed clipping lives in :mod:`opaque.clipping`; DP-FTRL requires fixed
sensitivity across steps (the single-shot MF privacy proof breaks under
adaptive / AUTO-S clipping).  Functional optimizers (including the
universal ``adamw`` that consumes private ``noisy_squared_grads`` streams)
live in :mod:`opaque.optimizers`.

Strategy and noise-state dataclasses (``BandMfStrategy``, ``BltStrategy``,
``BisrStrategy``, ``BsrStrategy``, ``IdentityStrategy``,
``LambdaCgdStrategy``, ``MfStrategy``, ``MFNoiseState``,
``SecondMomentMFNoiseState``) live in :mod:`opaque.dpftrl.noise.types`.
The cross-cutting ``SecondMomentNoiseOutput`` lives in :mod:`opaque.types`.

The :mod:`opaque.dpftrl.accounting` subpackage (DP-FTRL-specific privacy
accounting factories, requires ``opaque-accounting``) is **lazy-imported**:
``import opaque.dpftrl; opaque.dpftrl.accounting.band_mf(...)`` works, but
the underlying Rust PLD extension is only loaded on first attribute
access — so callers that only need noise / sampling do not pay the
extension's startup cost.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

from opaque.dpftrl import noise, sampling
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)

if TYPE_CHECKING:
    # Static type checkers see ``accounting`` as a real attribute; at
    # runtime it is loaded on first access via ``__getattr__`` below.
    from opaque.dpftrl import accounting as accounting

try:
    __version__ = _pkg_version("opaque-dpftrl")
except PackageNotFoundError:
    __version__ = "0.0.0"


_LAZY_SUBMODULES = frozenset({"accounting"})


def __getattr__(name: str):
    """PEP 562 lazy import for ``opaque.dpftrl.accounting``.

    Defers loading ``opaque.accounting`` (and its native Rust extension)
    until ``opaque.dpftrl.accounting`` is actually accessed.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(f"opaque.dpftrl.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'opaque.dpftrl' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Subpackages
    "accounting",
    "noise",
    "sampling",
    # Dispatchers
    "mf_noise",
    # Strategy factories
    "band_mf_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    # Samplers
    "BallsInBinsSampler",
    "BMinSepSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
