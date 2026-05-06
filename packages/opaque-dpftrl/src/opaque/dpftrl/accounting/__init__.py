"""DP-FTRL-specific accounting factories.

Re-exports the mechanism / amplification primitives that are scoped to
DP-FTRL (matrix-factorisation correlated-noise mechanisms with
band-aware subsampling) from the underlying :mod:`opaque.accounting`
implementation:

Mechanisms:

- :func:`band_mf` — banded matrix-factorisation Gaussian.
- :func:`blt` — buffered-linear-toeplitz Gaussian.
- :func:`bisr` — banded inverse square root Gaussian.
- :func:`bsr` — banded square root Gaussian.
- :func:`lambda_cgd` — DP-λCGD Gaussian.

Amplification:

- :func:`cyclic_poisson` — cyclic Poisson subsampling for BandMF.
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for BandMF.

Cross-cutting primitives (``balls_in_bins``, ``second_moment``,
composition, calibration) live at :mod:`opaque.accounting`.

Example::

    import opaque.accounting as acc
    import opaque.dpftrl.accounting as ftrl_acc

    step = ftrl_acc.cyclic_poisson(
        ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
        sample_rate=0.01,
    )
    eps = step.epsilon_at(1e-5)
"""

from opaque.accounting.amplification import b_min_sep, cyclic_poisson
from opaque.accounting.mechanisms import band_mf, bisr, blt, bsr, lambda_cgd

__all__ = [
    "band_mf",
    "blt",
    "bisr",
    "bsr",
    "lambda_cgd",
    "cyclic_poisson",
    "b_min_sep",
]
