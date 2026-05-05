"""Matrix Factorization noise mechanisms for correlated noise in DP-SGD.

Public API:

- :func:`mf_noise` — strategy-based dispatcher (SGD + Polyak momentum)
- ``mf_noise(..., second_moment=True, second_moment_strategy=...)`` — private
    second-moment stream for Adam-style optimizers

Strategy factories:

- :func:`band_mf_strategy`, :func:`blt_strategy`, :func:`bisr_strategy`, :func:`bsr_strategy`
- :func:`lambda_cgd_strategy`, :func:`identity_strategy`

Second-moment calibration helpers are re-exported from :mod:`opaque.core.noise`.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - Private second moments: https://arxiv.org/abs/2502.06597
"""

from opaque.core.noise import (
    DEFAULT_SECOND_MOMENT_OVERHEAD,
    second_moment_joint_sensitivity,
    second_moment_noise_scale,
    second_moment_stddevs,
)
from opaque.types import SecondMomentNoiseOutput

from .band_mf import BandMfStrategy, band_mf_strategy
from .bisr import BisrStrategy, bisr_strategy
from .bsr import BsrStrategy, bsr_strategy
from .blt import BltStrategy, blt_strategy
from .dispatcher import MfStrategy, mf_noise
from .identity import IdentityStrategy, identity_strategy
from .lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy
from .second_moment import SecondMomentMFNoiseState
from ._engine import MFNoiseState

__all__ = [
    # Dispatchers
    "mf_noise",
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
    # Second-moment helpers
    "DEFAULT_SECOND_MOMENT_OVERHEAD",
    "second_moment_joint_sensitivity",
    "second_moment_noise_scale",
    "second_moment_stddevs",
    # State / output
    "MFNoiseState",
    "SecondMomentMFNoiseState",
    "SecondMomentNoiseOutput",
]
