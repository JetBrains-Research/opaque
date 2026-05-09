"""DP-FTRL-specific accounting factories.

Mechanism and amplification primitives scoped to DP-FTRL (matrix-
factorisation correlated-noise mechanisms with band-aware subsampling):

Mechanisms (in :mod:`opaque.dpftrl.accounting.mechanisms`):

- :func:`band_mf` — banded matrix-factorisation Gaussian.
- :func:`blt` — buffered-linear-toeplitz Gaussian.
- :func:`bisr` — banded inverse square root Gaussian.
- :func:`bsr` — banded square root Gaussian.
- :func:`lambda_cgd` — DP-λCGD Gaussian.

Amplification (in :mod:`opaque.dpftrl.accounting.amplification`):

- :func:`cyclic_poisson` — cyclic Poisson subsampling for BandMF.
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for BandMF.
- :func:`balls_in_bins` — total multi-epoch cost under fixed-bin sampling
  for correlated-noise mechanisms (BLT/λCGD/BISR/BSR).

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`.

**MF identity baseline** (noise: :func:`~opaque.dpftrl.noise.identity_strategy`)
implements DP-SGD–style uncorrelated Gaussian noise through the MF API. Its
privacy analysis uses **Gaussian + Poisson subsampling per step**, composed for
the full run—the same accountant pattern as ``examples/train_dp_ftrl.py``
(``opaque.dpsgd.accounting``: ``poisson(gaussian(nm), …) * total_steps``). The
**correlated** MF mechanism factories listed above cover BLT/BandMF/λ-CGD/etc.;
they are not interchangeable with that baseline.

**Do not confuse** with :func:`~opaque.accounting.identity` — that object is the
**composition algebra** identity (approximately ε=0), not MF
``identity_strategy``.

Example::

    import opaque.accounting as acc
    import opaque.dpftrl.accounting as ftrl_acc

    step = ftrl_acc.cyclic_poisson(
        ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
        sample_rate=0.01,
    )
    eps = step.epsilon_at(1e-5)
"""

from opaque.dpftrl.accounting.amplification import (
    b_min_sep,
    balls_in_bins,
    cyclic_poisson,
)
from opaque.dpftrl.accounting.mechanisms import band_mf, bisr, blt, bsr, lambda_cgd

__all__ = [
    "band_mf",
    "blt",
    "bisr",
    "bsr",
    "lambda_cgd",
    "cyclic_poisson",
    "b_min_sep",
    "balls_in_bins",
]
