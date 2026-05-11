"""DP-FTRL-specific accounting factories façade.

**DP-FTRL accountants describe whole training processes.** Unlike DP-SGD
where a per-step factory composes externally with ``* num_steps``, every
factory here returns a :class:`~opaque.accounting.DpProcess` representing
the full training run. Length lives on amplification factories as
``n_steps``. Mechanism dataclasses are length-free; their ``sensitivity``,
``coefficients`` (BandMF), or ``gram_matrix`` (BLT/BISR/BSR/λCGD) fields
capture the strategy's structural decomposition once, computed at
strategy-construction time.

Mechanisms (in :mod:`opaque.dpftrl.accounting.mechanisms`):

- :func:`band_mf` — banded matrix-factorisation Gaussian.
- :func:`blt` — buffered-linear-toeplitz Gaussian.
- :func:`bisr` — banded inverse square root Gaussian.
- :func:`bsr` — banded square root Gaussian.
- :func:`lambda_cgd` — DP-λCGD Gaussian.
- :func:`identity_mf` — uncorrelated (identity-encoder) sensitivity-1 Gaussian.

Amplification (in :mod:`opaque.dpftrl.accounting.amplification`):

- :func:`poisson` — Poisson subsampling. Accepts ``BandMf`` (bands read
  from ``len(inner.coefficients)``) or ``IdentityMf`` (bands ≡ 1).
  Required keyword: ``n_steps`` (total training rounds).
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for ``BandMf``.
  Required keywords: ``n_steps``, ``p0``.
- :func:`balls_in_bins` — total privacy cost under fixed-partition
  Balls-in-Bins sampling. Required keywords: ``num_bins``, ``n_steps``
  (must be a positive multiple of ``num_bins``).

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`.

The amplification dataclass is named ``CyclicPoisson`` (rather than ``Poisson``)
to avoid a class-name collision with
:class:`opaque.dpsgd.accounting.amplification.Poisson` in the serialization
registry. The user-facing factory is still :func:`poisson`.

**Do not confuse** with :func:`~opaque.accounting.identity` — that object
is the **composition algebra** identity (approximately ε=0), not MF
``identity_mf_strategy``.

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

from opaque.api.accounting.dpftrl import (
    at_step,
    b_min_sep,
    balls_in_bins,
    band_mf,
    bisr,
    blt,
    bsr,
    identity_mf,
    lambda_cgd,
    poisson,
)

__all__ = [
    "band_mf",
    "blt",
    "bisr",
    "bsr",
    "identity_mf",
    "lambda_cgd",
    "poisson",
    "b_min_sep",
    "balls_in_bins",
    "at_step",
]
