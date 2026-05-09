"""DP-SGD-specific accounting factories.

Mechanism and amplification primitives scoped to DP-SGD (independent-noise
per-step Gaussian + subsampling):

Mechanisms (in :mod:`opaque.dpsgd.accounting.mechanisms`):

- :func:`gaussian` — base Gaussian mechanism.
- :func:`adaclip` — adaptive-clipping transformation.

Amplification (in :mod:`opaque.dpsgd.accounting.amplification`):

- :func:`poisson` — Poisson subsampling.  Set ``truncated_batch_size``
  and ``dataset_size`` together for the truncated-Poisson production form.
- :func:`parallel_poisson` — Poisson subsampling under parallel workers.

Each amplification factory returns a **per-step** :class:`DpProcess`;
compose externally with ``* num_steps`` for full-training privacy.

Cross-cutting primitives (composition, calibration) live at
:mod:`opaque.accounting`. DP-FTRL helpers such as :func:`balls_in_bins`
live in :mod:`opaque.dpftrl.accounting`.

Example::

    import opaque.accounting as acc
    import opaque.dpsgd.accounting as dpsgd_acc

    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    eps = training.epsilon_at(1e-5)
"""

from opaque.dpsgd.accounting.amplification import (
    parallel_poisson,
    poisson,
)
from opaque.dpsgd.accounting.mechanisms import adaclip, gaussian

__all__ = [
    "gaussian",
    "adaclip",
    "poisson",
    "parallel_poisson",
]
