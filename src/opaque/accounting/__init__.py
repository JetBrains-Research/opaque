"""Functional privacy accounting using dp_accounting + riskcal.

This module provides a clean, functional API for differential privacy accounting
during DP-SGD training. It uses Google's dp_accounting library (PLD) directly as
the state and riskcal for alpha/beta queries.

The design is purely functional:
- Immutable state (PLD is already immutable)
- Pure functions (all methods return new PLD objects)
- Composable (state can be saved, cached, checkpointed)

Example:
    >>> import opaque.accounting as acc
    >>>
    >>> # Initialize state
    >>> state = acc.create()
    >>>
    >>> # DP-SGD with truncated Poisson sampling
    >>> state = acc.compose_truncated_poisson_gaussian(
    ...     state,
    ...     noise_multiplier=1.1,
    ...     sample_rate=0.01,
    ...     truncated_batch_size=100,
    ...     dataset_size=10000,
    ...     count=1000
    ... )
    >>>
    >>> # Traditional (ε, δ) query
    >>> epsilon = acc.get_epsilon(state, delta=1e-5)
    >>> print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")
    >>>
    >>> # Modern alpha/beta query (operational risk)
    >>> beta = acc.get_beta(state, alpha=0.01)
    >>> print(f"At 1% FPR, FNR = {beta:.3f}")
"""

# Re-export PLD as PrivacyState for type annotations
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution as PrivacyState,
)
from dp_accounting.privacy_accountant import NeighboringRelation

# Calibration (using riskcal.calibration.core primitives)
from opaque.accounting.calibration import (
    # Core primitives
    CalibrationConfig,
    CalibrationResult,
    CalibrationTarget,
    PrivacyEvaluator,
    PrivacyMetrics,
    calibrate_parameter,
    # Evaluator factories
    create_dpsgd_epsilon_evaluator,
    create_dpsgd_advantage_evaluator,
    create_dpsgd_beta_evaluator,
    # Calibration functions
    find_noise_multiplier_for_epsilon_delta,
    find_noise_multiplier_for_advantage,
    find_noise_multiplier_for_err_rates,
    # Query functions
    get_epsilon_for_dpsgd,
    get_advantage_for_dpsgd,
    get_beta_for_dpsgd,
)

# Composition functions
from opaque.accounting.composition import (
    compose_fixed_batch,
    compose_poisson_gaussian,
    compose_sampled_gaussian,
    compose_truncated_poisson_gaussian,
    create,
)

# Query functions
from opaque.accounting.queries import (
    get_advantage,
    get_beta,
    get_delta,
    get_epsilon,
    get_privacy_curve,
)

__all__ = [
    # State type (PLD)
    "PrivacyState",
    # DP configuration
    "NeighboringRelation",
    # Composition
    "create",
    "compose_poisson_gaussian",
    "compose_sampled_gaussian",
    "compose_fixed_batch",
    "compose_truncated_poisson_gaussian",
    # Queries
    "get_epsilon",
    "get_delta",
    "get_beta",
    "get_advantage",
    "get_privacy_curve",
    # Calibration - core primitives
    "PrivacyEvaluator",
    "PrivacyMetrics",
    "CalibrationTarget",
    "CalibrationConfig",
    "CalibrationResult",
    "calibrate_parameter",
    # Calibration - evaluator factories
    "create_dpsgd_epsilon_evaluator",
    "create_dpsgd_advantage_evaluator",
    "create_dpsgd_beta_evaluator",
    # Calibration - functions
    "find_noise_multiplier_for_epsilon_delta",
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    # Calibration - query functions
    "get_epsilon_for_dpsgd",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
]
