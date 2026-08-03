"""DP-FTRL-specific accounting factories façade.

**DP-FTRL accountants describe whole training processes.** Unlike DP-SGD
where a per-step factory composes externally with ``* num_steps``, every
factory here returns a :class:`~opaque.accounting.DpProcess` representing
the full training run.  Length lives on amplification factories as
``n_steps``.

The accounting mechanism is :class:`MfGaussian` — a thin wrapper over
``(noise_multiplier, strategy)``.  The strategy (from
:mod:`opaque.dpftrl.noise`) carries the structural decomposition once;
amplifications dispatch on its type at PLD time.

Mechanism factory (in :mod:`opaque.dpftrl.accounting.mechanisms`):

- :func:`mf_gaussian` — single-argument-pair MF Gaussian factory:
  ``mf_gaussian(nm, strategy)``.

Amplification (in :mod:`opaque.dpftrl.accounting.amplification`):

- :func:`poisson` — Poisson subsampling.  Accepts MfGaussian wrapping a
  ``BandMfStrategy`` (bands read from ``len(coefficients)``) or
  ``IdentityStrategy`` (bands ≡ 1).  Required keyword: ``n_steps``.
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for
  ``BandMfStrategy``.  Required keywords: ``n_steps``, ``p0``.
- :func:`balls_in_bins` — total privacy cost under fixed-partition
  Balls-in-Bins sampling.  Required keywords: ``num_bins``, ``n_steps``
  (must be a positive multiple of ``num_bins``).

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`, including :func:`opaque.accounting.per_step` for
adapting a whole-horizon process to ``acc |= step`` loops.

The amplification dataclass is named ``CyclicPoisson`` (rather than ``Poisson``)
to avoid a class-name collision with
:class:`opaque.dpsgd.accounting.amplification.Poisson` in the serialization
registry.  The user-facing factory is still :func:`poisson`.

**Do not confuse** with :func:`~opaque.accounting.identity` — that object
is the **composition algebra** identity (approximately ε=0), not MF
``identity_strategy``.

Example::

    import opaque.accounting as acc
    import opaque.dpftrl.accounting as ftrl_acc
    from opaque.dpftrl.noise import band_mf_strategy, blt_strategy

    band_s = band_mf_strategy(bands=10)
    step = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, band_s), sample_rate=0.01, n_steps=1000,
    )
    eps = step.epsilon_at(1e-5)
"""

from opaque.api.accounting.core.composition._per_step import (
    per_step as per_step,
)
from opaque.api.accounting.dpftrl import (
    b_min_sep,
    balls_in_bins,
    mf_gaussian,
    poisson,
)
from opaque.accounting import per_step

__all__ = [
    "b_min_sep",
    "balls_in_bins",
    "mf_gaussian",
    "poisson",
]
