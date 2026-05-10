"""Matrix-factorization noise mechanisms façade for correlated noise in DP-FTRL.

Public API:

- :func:`mf_noise` — strategy-based dispatcher (SGD + Polyak momentum).
- ``mf_noise(..., second_moment_strategy=...)`` — optional paired stream
  for private second moments (Adam-style optimizers); requires
  ``second_moment_strategy`` at construction and
  ``SecondMomentClippingOutput`` at runtime.

Strategy factories:

- :func:`band_mf_strategy`, :func:`blt_strategy`, :func:`bisr_strategy`,
  :func:`bsr_strategy`, :func:`lambda_cgd_strategy`,
  :func:`identity_mf_strategy`.

Strategy types and noise state classes (``BandMfStrategy``, ``BltStrategy``,
``BisrStrategy``, ``BsrStrategy``, ``IdentityMfStrategy``,
``LambdaCgdStrategy``, ``MfStrategy``, ``MFNoiseState``,
``SecondMomentMFNoiseState``) live in :mod:`opaque.dpftrl.noise.types`.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - Private second moments: https://arxiv.org/abs/2502.06597
"""

from opaque.api.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_mf_strategy,
    lambda_cgd_strategy,
    mf_noise,
)

__all__ = [
    "mf_noise",
    "band_mf_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "blt_strategy",
    "identity_mf_strategy",
    "lambda_cgd_strategy",
]
