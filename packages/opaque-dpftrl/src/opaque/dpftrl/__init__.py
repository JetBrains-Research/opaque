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
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

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

try:
    __version__ = _pkg_version("opaque-dpftrl")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Subpackages
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
