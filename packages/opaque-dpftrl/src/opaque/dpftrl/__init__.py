"""Opaque DP-FTRL: correlated (matrix-factorization) noise mechanisms.

Strategies (BLT, BSR, BiSR, band-MF, λ-CGD, identity) + DP-FTRL-specific
participation samplers (b-min-sep, cyclic Poisson, balls-in-bins, sequential).

Fixed clipping lives in :mod:`opaque.clipping`; DP-FTRL requires fixed
sensitivity across steps (the single-shot MF privacy proof breaks under
adaptive / AUTO-S clipping).  Functional optimizers (including the
universal ``adamw`` that consumes private ``noisy_squared_grads`` streams)
live in :mod:`opaque.optimizers`.

Strategy data classes (``BandMfStrategy``, ``BltStrategy``, …) are importable
from this module for type annotations but are not part of ``__all__`` — the
public surface is functional (strategy factory functions + ``mf_noise`` /
``mf_noise(..., second_moment=True)`` dispatchers).
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.dpftrl import noise, sampling
from opaque.dpftrl.noise import BandMfStrategy as BandMfStrategy
from opaque.dpftrl.noise import BisrStrategy as BisrStrategy
from opaque.dpftrl.noise import BltStrategy as BltStrategy
from opaque.dpftrl.noise import BsrStrategy as BsrStrategy
from opaque.dpftrl.noise import IdentityStrategy as IdentityStrategy
from opaque.dpftrl.noise import LambdaCgdStrategy as LambdaCgdStrategy
from opaque.dpftrl.noise import MFNoiseState as MFNoiseState
from opaque.dpftrl.noise import MfStrategy as MfStrategy
from opaque.dpftrl.noise import SecondMomentMFNoiseState as SecondMomentMFNoiseState
from opaque.dpftrl.noise import SecondMomentNoiseOutput as SecondMomentNoiseOutput
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
    # State / output
    "MFNoiseState",
    "SecondMomentMFNoiseState",
    "SecondMomentNoiseOutput",
]
