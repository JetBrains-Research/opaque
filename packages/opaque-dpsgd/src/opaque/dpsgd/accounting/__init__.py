"""DP-SGD-specific accounting factories.

Re-exports the mechanism / transformation / amplification primitives that
are scoped to DP-SGD (independent-noise per-step Gaussian + subsampling)
from the underlying :mod:`opaque.accounting` implementation:

- :func:`gaussian` — base Gaussian mechanism for per-step independent noise.
- :func:`adaclip` — adaptive-clipping transformation (folds an extra
  Gaussian for clip-quantile estimation into the effective noise).
- :func:`poisson`, :func:`truncated_poisson`, :func:`parallel_poisson` —
  per-step Poisson subsampling amplifications.

Cross-cutting primitives (``balls_in_bins``, ``second_moment``,
composition, calibration) live at :mod:`opaque.accounting`.

Example::

    import opaque.accounting as acc
    import opaque.dpsgd.accounting as dpsgd_acc

    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    eps = training.epsilon_at(1e-5)
"""

from opaque.accounting.amplification import (
    parallel_poisson,
    poisson,
    truncated_poisson,
)
from opaque.accounting.mechanisms import gaussian
from opaque.accounting.transformations import adaclip

__all__ = [
    "gaussian",
    "adaclip",
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
]
