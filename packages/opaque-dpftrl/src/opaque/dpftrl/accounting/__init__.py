"""DP-FTRL accounting factories for whole training processes.

Unlike DP-SGD, where a per-step factory composes externally with the number of
steps, these amplification factories return a
:class:`~opaque.accounting.DpProcess` for the complete training run. The
participation horizon is therefore explicit: bare :func:`mf_gaussian` calls
require ``n_steps``, while amplification factories own that context and take an
inner ``MfGaussian`` with ``n_steps=1``.

The strategy (from :mod:`opaque.dpftrl.noise`) carries the structural
decomposition. Amplifications dispatch on its type when constructing the PLD:

- :func:`mf_gaussian` creates an unamplified MF Gaussian process for an explicit
  participation horizon.
- :func:`poisson` applies Poisson subsampling. It accepts an ``MfGaussian``
  wrapping a ``BandMfStrategy`` or ``IdentityStrategy`` and requires
  ``n_steps``.
- :func:`b_min_sep` provides warm-start b-min-sep Monte Carlo accounting for a
  ``BandMfStrategy`` and requires ``n_steps`` and ``p0``.
- :func:`balls_in_bins` returns the total privacy cost under fixed-partition
  Balls-in-Bins sampling and requires ``num_bins`` and ``n_steps``.

Cross-cutting composition and calibration live in :mod:`opaque.accounting`.
Its :func:`opaque.accounting.identity` is the composition identity, not the
matrix-factorization ``identity_strategy``. The amplification dataclass is
named ``CyclicPoisson`` to avoid a serialization-registry collision with the
DP-SGD ``Poisson`` class; its public factory remains :func:`poisson`.

Example::

    import opaque.dpftrl.accounting as ftrl_acc
    from opaque.dpftrl.noise import band_mf_strategy

    band_s = band_mf_strategy(bands=10)
    process = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, band_s, n_steps=1),
        sample_rate=0.01,
        n_steps=1000,
    )
    eps = process.epsilon_at(1e-5)
"""

from opaque.api.accounting.dpftrl import (
    b_min_sep,
    balls_in_bins,
    mf_gaussian,
    poisson,
)

__all__ = [
    "b_min_sep",
    "balls_in_bins",
    "mf_gaussian",
    "poisson",
]
