"""Matrix Factorization noise mechanisms for correlated noise in DP-SGD.

Public API:

- :func:`mf_noise` — strategy-based dispatcher (SGD + Polyak momentum)
- ``mf_noise(..., second_moment_strategy=...)`` — optional paired stream for
  private second moments (Adam-style optimizers); requires
  ``second_moment_strategy`` at construction and
  ``SecondMomentClippingOutput`` at runtime

Strategy factories:

- :func:`band_mf_strategy`, :func:`blt_strategy`, :func:`bisr_strategy`,
  :func:`bsr_strategy`, :func:`lambda_cgd_strategy`, :func:`identity_strategy`

Strategy types and noise state classes (``BandMfStrategy``, ``BltStrategy``,
``BisrStrategy``, ``BsrStrategy``, ``IdentityStrategy``,
``LambdaCgdStrategy``, ``MfStrategy``, ``MFNoiseState``,
``SecondMomentMFNoiseState``) live in :mod:`opaque.dpftrl.noise.types`.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - Private second moments: https://arxiv.org/abs/2502.06597
"""

from ._band_mf import band_mf_strategy
from ._bisr import bisr_strategy
from ._blt import blt_strategy
from ._bsr import bsr_strategy
from ._dispatcher import mf_noise
from ._identity import identity_strategy
from ._lambda_cgd import lambda_cgd_strategy

import opaque.dpftrl.noise._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "mf_noise",
    "band_mf_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
]
