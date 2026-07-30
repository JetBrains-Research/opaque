"""Differential privacy accounting using Privacy Loss Distributions (PLD).

Cross-cutting accounting surface — composition, calibration, generic
mechanisms (``identity``, ``nonprivate``, ``eps_delta``).

Algorithm-specific factories live in their respective packages
(``opaque-dpsgd`` / ``opaque-dpftrl``):

- :mod:`opaque.dpsgd.accounting` — ``gaussian``, ``adaclip``, ``poisson``,
  ``truncated_poisson``, ``parallel_poisson``.
- :mod:`opaque.dpftrl.accounting` — ``band_mf``, ``blt``, ``bisr``,
  ``bsr``, ``lambda_cgd``, ``identity_mf``, ``poisson``, ``b_min_sep``,
  ``balls_in_bins``.

Implementation uses Google's PLD accounting via the ``opaque-accounting``
Rust crate (PyO3 bindings).

Example (requires ``opaque-dpsgd`` in the environment)::

    import opaque.accounting as acc
    import opaque.dpsgd.accounting as dpsgd_acc

    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)
"""

from opaque.api.accounting.core import (
    Accountant,
    __version__,
    advantage_budget,
    amplification,
    beta_budget,
    cached,
    calibrate,
    calibration,
    compose,
    composition,
    delta_budget,
    discretization,
    eps_delta,
    epsilon_budget,
    get_discretization,
    identity,
    mechanisms,
    nonprivate,
    repeat,
    risk_budget,
    set_discretization,
)

__all__ = [
    # Accountant
    "Accountant",
    "__version__",
    "advantage_budget",
    # Submodules
    "amplification",
    "beta_budget",
    "cached",
    "calibrate",
    "calibration",
    "compose",
    "composition",
    "delta_budget",
    "discretization",
    # Generic mechanisms
    "eps_delta",
    # Calibration / budgets
    "epsilon_budget",
    "get_discretization",
    "identity",
    "mechanisms",
    "nonprivate",
    # Composition
    "repeat",
    "risk_budget",
    # Discretization
    "set_discretization",
]
