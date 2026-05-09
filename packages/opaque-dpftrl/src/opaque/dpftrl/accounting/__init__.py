"""DP-FTRL-specific accounting factories.

**DP-FTRL accountants describe whole training processes.** Unlike DP-SGD
where a per-step factory composes externally with ``* num_steps``, every
factory here returns a :class:`~opaque.accounting.DpProcess` representing
the full training run.  Length lives on amplification factories as
``n_steps``.  Mechanism dataclasses are length-free; their ``sensitivity``,
``coefficients`` (BandMF), or ``gram_matrix`` (BLT/BISR/BSR/λCGD) fields
capture the strategy's structural decomposition once, computed at
strategy-construction time.

Mechanisms (in :mod:`opaque.dpftrl.accounting.mechanisms`):

- :func:`band_mf` — banded matrix-factorisation Gaussian.
- :func:`blt` — buffered-linear-toeplitz Gaussian.
- :func:`bisr` — banded inverse square root Gaussian.
- :func:`bsr` — banded square root Gaussian.
- :func:`lambda_cgd` — DP-λCGD Gaussian.
- :func:`mf_identity` — uncorrelated (identity-encoder) sensitivity-1 Gaussian.

Amplification (in :mod:`opaque.dpftrl.accounting.amplification`):

- :func:`poisson` — Poisson subsampling.  Accepts ``BandMf`` (bands read
  from ``len(inner.coefficients)``) or ``IdentityMf`` (bands ≡ 1).
  Required keyword: ``n_steps`` (total training rounds).
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for ``BandMf``.
  Required keywords: ``n_steps``, ``p0``.
- :func:`balls_in_bins` — total privacy cost under fixed-partition
  Balls-in-Bins sampling.  Required keywords: ``num_bins``, ``n_steps``
  (must be a positive multiple of ``num_bins``).

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`.

Mechanism and amplification **dataclasses** (e.g.
:class:`~opaque.dpftrl.accounting.types.IdentityMf`,
:class:`~opaque.dpftrl.accounting.types.BandMf`,
:class:`~opaque.dpftrl.accounting.types.MfPoisson`) are exported from
:mod:`opaque.dpftrl.accounting.types`, not re-imported at this package root.

The amplification dataclass is named ``MfPoisson`` (rather than ``Poisson``)
to avoid a class-name collision with
:class:`opaque.dpsgd.accounting.amplification.Poisson` in the serialization
registry.  The user-facing factory is still :func:`poisson`.

**MF identity baseline** pairs :func:`~opaque.dpftrl.noise.identity_strategy`
with :func:`~opaque.dpftrl.accounting.mechanisms.mf_identity` (sensitivity-1
Gaussian).  Realistic FTRL training composes this through one of the
amplification factories — e.g. ``poisson(mf_identity(nm), sample_rate=p,
n_steps=T)`` or ``balls_in_bins(mf_identity(nm), num_bins=k, n_steps=k*E)``.
The :func:`poisson` factory is the FTRL-native Poisson amplification; it
is **independent** of :func:`opaque.dpsgd.accounting.poisson`, which has
the per-step shape used by DP-SGD's external composition.

**Do not confuse** with :func:`~opaque.accounting.identity` — that object is the
**composition algebra** identity (approximately ε=0), not MF
``identity_strategy``.

Example::

    import opaque.accounting as acc
    import opaque.dpftrl.accounting as ftrl_acc

    step = ftrl_acc.poisson(
        ftrl_acc.band_mf(1.0, sensitivity=1.0,
                         coefficients=strategy.coefficients),
        sample_rate=0.01,
        n_steps=1000,
    )
    eps = step.epsilon_at(1e-5)
"""

from opaque.dpftrl.accounting.amplification import (
    b_min_sep,
    balls_in_bins,
    poisson,
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
    "poisson",
    "b_min_sep",
    "balls_in_bins",
]
