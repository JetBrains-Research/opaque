"""DP-FTRL-specific accounting factories.

Mechanism and amplification primitives scoped to DP-FTRL (matrix-
factorisation correlated-noise mechanisms with band-aware subsampling):

Mechanisms (in :mod:`opaque.dpftrl.accounting.mechanisms`):

- :func:`band_mf` — banded matrix-factorisation Gaussian.
- :func:`blt` — buffered-linear-toeplitz Gaussian.
- :func:`bisr` — banded inverse square root Gaussian.
- :func:`bsr` — banded square root Gaussian.
- :func:`lambda_cgd` — DP-λCGD Gaussian.
- :func:`mf_identity` — uncorrelated (identity-encoder) sensitivity-1 Gaussian.
  Use as the inner of an FTRL amplification factory (or compose stand-alone
  with ``mf_identity(nm) * num_steps`` for unsubsampled training).

Amplification (in :mod:`opaque.dpftrl.accounting.amplification`):

- :func:`cyclic_poisson` — cyclic Poisson subsampling.  Accepts ``BandMf``
  (cycle count from ``num_groups``) or ``IdentityMf`` (cycle count from
  ``num_steps`` keyword; every step is its own group).
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for BandMF.
- :func:`balls_in_bins` — total multi-epoch cost under fixed-bin sampling
  for correlated-noise mechanisms (BLT/λCGD/BISR/BSR) and tight identity
  reduction for ``IdentityMf``.

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`.

Mechanism **dataclasses** (e.g. :class:`~opaque.dpftrl.accounting.types.IdentityMf`,
:class:`~opaque.dpftrl.accounting.types.BandMf`) are exported from
:mod:`opaque.dpftrl.accounting.types`, not re-imported at this package root.

**MF identity baseline** pairs :func:`~opaque.dpftrl.noise.identity_strategy`
with :func:`~opaque.dpftrl.accounting.mechanisms.mf_identity` (sensitivity-1
Gaussian).  Realistic FTRL training composes this through one of the FTRL
amplification factories — e.g.
``cyclic_poisson(mf_identity(nm), sample_rate=p, num_steps=T)`` for
per-step Poisson, or ``balls_in_bins(mf_identity(nm), num_bins=k, num_epochs=E)``
for fixed-partition (with the tight identity-aware reduction inside
``balls_in_bins``).  DP-FTRL does **not** expose a generic ``poisson(...)``
factory; ``cyclic_poisson`` is the FTRL-native per-step Poisson amplification.

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
from opaque.dpftrl.accounting.mechanisms import (
    band_mf,
    bisr,
    blt,
    bsr,
    lambda_cgd,
    mf_identity,
)

__all__ = [
    "band_mf",
    "blt",
    "bisr",
    "bsr",
    "lambda_cgd",
    "mf_identity",
    "cyclic_poisson",
    "b_min_sep",
    "balls_in_bins",
]
