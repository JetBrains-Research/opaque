"""Opaque DP-FTRL: matrix-factorization noise mechanisms for DP-FTRL.

This package provides correlated-noise mechanisms (BLT, BSR, BiSR,
band-MF, JME, lambda-CGD, identity) together with the
``AdamW-JME`` optimizer and the DP-FTRL-specific participation
samplers (b-min-sep, cyclic Poisson, balls-in-bins, sequential).

Partition policy: DP-FTRL-specific code lives here; shared primitives
(fixed clipping, RNG, pytree utilities) live in :mod:`opaque.core`.
"""

from opaque.dpftrl import noise, optimizers, sampling
from opaque.dpftrl.noise import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    BsrStrategy,
    IdentityStrategy,
    JmeNoiseOutput,
    JmeNoiseState,
    LambdaCgdStrategy,
    MFNoiseState,
    MfStrategy,
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    jme_joint_sensitivity,
    jme_lambda,
    jme_noise,
    jme_second_moment_stddev,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # Subpackages
    "noise",
    "optimizers",
    "sampling",
    # Dispatchers
    "mf_noise",
    "jme_noise",
    "MfStrategy",
    # Strategy types & factories
    "BandMfStrategy",
    "band_mf_strategy",
    "BisrStrategy",
    "bisr_strategy",
    "BsrStrategy",
    "bsr_strategy",
    "BltStrategy",
    "blt_strategy",
    "IdentityStrategy",
    "identity_strategy",
    "LambdaCgdStrategy",
    "lambda_cgd_strategy",
    # JME helpers
    "jme_lambda",
    "jme_joint_sensitivity",
    "jme_second_moment_stddev",
    # State / output
    "MFNoiseState",
    "JmeNoiseState",
    "JmeNoiseOutput",
    # Samplers
    "BallsInBinsSampler",
    "BMinSepSampler",
    "CyclicPoissonSampler",
    "SequentialBatchSampler",
]
