"""Matrix Factorization noise mechanisms for correlated noise in DP-SGD.

Public API:

- :func:`mf_noise` — strategy-based dispatcher (SGD + Polyak momentum)
- :func:`mf_noise_jme` — JME wrapper for DP-Adam (two streams, one budget)

Strategy factories:

- :func:`band_mf_strategy`, :func:`blt_strategy`, :func:`bisr_strategy`
- :func:`lambda_cgd_strategy`, :func:`identity_strategy`

JME calibration helpers:

- :func:`jme_lambda`, :func:`jme_joint_sensitivity`, :func:`jme_second_moment_stddev`

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - JME: https://arxiv.org/abs/2502.06597
"""

from .band_mf import BandMfStrategy, band_mf_strategy
from .bisr import BisrStrategy, bisr_strategy
from .blt import BltStrategy, blt_strategy
from .dispatcher import MfStrategy, mf_noise
from .identity import IdentityStrategy, identity_strategy
from .jme import (
    JmeNoiseState,
    jme_joint_sensitivity,
    jme_lambda,
    jme_second_moment_stddev,
    mf_noise_jme,
)
from .lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy
from ._engine import MFNoiseState

__all__ = [
    # Dispatchers
    "mf_noise",
    "mf_noise_jme",
    "MfStrategy",
    # Strategy types & factories
    "BandMfStrategy",
    "band_mf_strategy",
    "BisrStrategy",
    "bisr_strategy",
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
    # State
    "MFNoiseState",
    "JmeNoiseState",
]
