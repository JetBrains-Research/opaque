"""Calibration functions using riskcal.calibration.core primitives.

Provides PLD-based calibration for Poisson-sampled DP-SGD with three metrics:
- (ε, δ)-differential privacy
- (α, β) error rates
- advantage (f-DP)

This module follows riskcal's design pattern:
1. Create evaluator (factory function returning PrivacyEvaluator)
2. Define target (CalibrationTarget with kind and values)
3. Configure search (CalibrationConfig with bounds and tolerances)
4. Calibrate (calibrate_parameter returns CalibrationResult)
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
# Re-export riskcal core primitives
from riskcal.calibration.core import (
    CalibrationConfig,
    CalibrationResult,
    CalibrationTarget,
    PrivacyEvaluator,
    PrivacyMetrics,
    calibrate_parameter,
)

# Import opaque.accounting for our custom evaluators
import opaque.accounting as acc


def create_dpsgd_epsilon_evaluator(
    num_steps: int,
    target_delta: float,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> PrivacyEvaluator:
    """Create epsilon-delta evaluator for DP-SGD.

    Uses opaque.accounting for PLD composition with different sampling methods.

    Args:
        num_steps: Number of DP-SGD steps.
        target_delta: Delta parameter for epsilon computation.
        sampling_method: Sampling method ("poisson", "truncated_poisson", "fixed_batch").
        sample_rate: Poisson sampling probability (required for "poisson" and "truncated_poisson").
        batch_size: Batch size (required for "fixed_batch").
        dataset_size: Dataset size (required for "fixed_batch" and "truncated_poisson").
        truncated_batch_size: Max batch size (required for "truncated_poisson").
        grid_step: PLD discretization interval.

    Returns:
        PrivacyEvaluator that maps noise_multiplier → PrivacyMetrics.

    Example:
        >>> # Poisson sampling
        >>> evaluator = create_dpsgd_epsilon_evaluator(
        ...     num_steps=1000, target_delta=1e-5,
        ...     sampling_method="poisson", sample_rate=0.01
        ... )
        >>> # Fixed batch
        >>> evaluator = create_dpsgd_epsilon_evaluator(
        ...     num_steps=1000, target_delta=1e-5,
        ...     sampling_method="fixed_batch", batch_size=32, dataset_size=10000
        ... )
    """
    # Validate parameters based on sampling method
    if sampling_method == "poisson":
        if sample_rate is None:
            raise ValueError("sample_rate required for poisson sampling")
    elif sampling_method == "fixed_batch":
        if batch_size is None or dataset_size is None:
            raise ValueError("batch_size and dataset_size required for fixed_batch")
    elif sampling_method == "truncated_poisson":
        if sample_rate is None or dataset_size is None or truncated_batch_size is None:
            raise ValueError(
                "sample_rate, dataset_size, and truncated_batch_size "
                "required for truncated_poisson"
            )

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate epsilon-delta for given noise."""
        # Create state
        state = acc.create(discretization_interval=grid_step)

        # Compose based on sampling method
        if sampling_method == "poisson":
            state = acc.compose_poisson_gaussian(
                state,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                count=num_steps,
            )
        elif sampling_method == "fixed_batch":
            state = acc.compose_sampled_gaussian(
                state,
                noise_multiplier=noise_multiplier,
                batch_size=batch_size,
                dataset_size=dataset_size,
                count=num_steps,
            )
        elif sampling_method == "truncated_poisson":
            state = acc.compose_truncated_poisson_gaussian(
                state,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size,
                count=num_steps,
            )

        # Get epsilon at target delta
        epsilon = acc.get_epsilon(state, delta=target_delta)

        return PrivacyMetrics(epsilon=epsilon, delta=target_delta)

    return evaluator


def find_noise_multiplier_for_epsilon_delta(
    epsilon: float,
    delta: float,
    num_steps: int,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
    eps_tol: float = 1e-2,
    noise_min: float = 0.1,
    noise_max: float = 50.0,
) -> float:
    """Calibrate noise_multiplier to achieve target (ε, δ)-DP.

    Uses opaque.accounting for tight PLD-based privacy analysis with different
    sampling methods.

    Args:
        epsilon: Target epsilon value.
        delta: Target delta value.
        num_steps: Number of DP-SGD steps.
        sampling_method: Sampling method ("poisson", "truncated_poisson", "fixed_batch").
        sample_rate: Poisson sampling probability (required for "poisson" and "truncated_poisson").
        batch_size: Batch size (required for "fixed_batch").
        dataset_size: Dataset size (required for "fixed_batch" and "truncated_poisson").
        truncated_batch_size: Max batch size (required for "truncated_poisson").
        grid_step: PLD discretization interval.
        eps_tol: Convergence tolerance for epsilon.
        noise_min: Lower bound for search.
        noise_max: Upper bound for search.

    Returns:
        Calibrated noise_multiplier.

    Example:
        >>> # Poisson sampling
        >>> noise = find_noise_multiplier_for_epsilon_delta(
        ...     epsilon=3.0, delta=1e-5, num_steps=1000,
        ...     sampling_method="poisson", sample_rate=0.01
        ... )
        >>> # Fixed batch
        >>> noise = find_noise_multiplier_for_epsilon_delta(
        ...     epsilon=3.0, delta=1e-5, num_steps=1000,
        ...     sampling_method="fixed_batch", batch_size=32, dataset_size=10000
        ... )
    """
    # Create evaluator
    evaluator = create_dpsgd_epsilon_evaluator(
        num_steps=num_steps,
        target_delta=delta,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        batch_size=batch_size,
        dataset_size=dataset_size,
        truncated_batch_size=truncated_batch_size,
        grid_step=grid_step,
    )

    # Define target
    target = CalibrationTarget(
        kind="epsilon_delta",
        epsilon=epsilon,
        delta=delta,
    )

    # Configure search
    config = CalibrationConfig(
        param_min=noise_min,
        param_max=noise_max,
        target_tol=eps_tol,
        increasing=False,  # Higher noise → lower epsilon
    )

    # Calibrate
    result = calibrate_parameter(
        evaluator,
        target,
        config,
        parameter_name="noise_multiplier",
    )

    return result.parameter_value


def get_epsilon_for_dpsgd(
    noise_multiplier: float,
    num_steps: int,
    delta: float,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> float:
    """Compute epsilon for DP-SGD with given parameters.

    Args:
        noise_multiplier: Noise scale parameter.
        num_steps: Number of DP-SGD steps.
        delta: Delta parameter for epsilon computation.
        sampling_method: Sampling method ("poisson", "truncated_poisson", "fixed_batch").
        sample_rate: Poisson sampling probability (required for "poisson" and "truncated_poisson").
        batch_size: Batch size (required for "fixed_batch").
        dataset_size: Dataset size (required for "fixed_batch" and "truncated_poisson").
        truncated_batch_size: Max batch size (required for "truncated_poisson").
        grid_step: PLD discretization interval.

    Returns:
        Epsilon value at target delta.

    Example:
        >>> # Poisson sampling
        >>> eps = get_epsilon_for_dpsgd(
        ...     noise_multiplier=1.1, num_steps=1000, delta=1e-5,
        ...     sampling_method="poisson", sample_rate=0.01
        ... )
        >>> # Fixed batch
        >>> eps = get_epsilon_for_dpsgd(
        ...     noise_multiplier=1.1, num_steps=1000, delta=1e-5,
        ...     sampling_method="fixed_batch", batch_size=32, dataset_size=10000
        ... )
    """
    evaluator = create_dpsgd_epsilon_evaluator(
        num_steps=num_steps,
        target_delta=delta,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        batch_size=batch_size,
        dataset_size=dataset_size,
        truncated_batch_size=truncated_batch_size,
        grid_step=grid_step,
    )
    metrics = evaluator(noise_multiplier)
    return metrics.epsilon


def create_dpsgd_advantage_evaluator(
    num_steps: int,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> PrivacyEvaluator:
    """Create advantage evaluator for DP-SGD.

    Args:
        num_steps: Number of DP-SGD steps.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.

    Returns:
        PrivacyEvaluator that maps noise_multiplier → PrivacyMetrics.
    """
    # Parameter validation (same as epsilon evaluator)
    if sampling_method == "poisson":
        if sample_rate is None:
            raise ValueError("sample_rate required for poisson sampling")
    elif sampling_method == "fixed_batch":
        if batch_size is None or dataset_size is None:
            raise ValueError("batch_size and dataset_size required for fixed_batch")
    elif sampling_method == "truncated_poisson":
        if sample_rate is None or dataset_size is None or truncated_batch_size is None:
            raise ValueError(
                "sample_rate, dataset_size, and truncated_batch_size "
                "required for truncated_poisson"
            )

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate advantage for given noise."""
        state = acc.create(discretization_interval=grid_step)

        # Compose based on sampling method
        if sampling_method == "poisson":
            state = acc.compose_poisson_gaussian(
                state, noise_multiplier=noise_multiplier,
                sample_rate=sample_rate, count=num_steps
            )
        elif sampling_method == "fixed_batch":
            state = acc.compose_sampled_gaussian(
                state, noise_multiplier=noise_multiplier,
                batch_size=batch_size, dataset_size=dataset_size, count=num_steps
            )
        elif sampling_method == "truncated_poisson":
            state = acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=noise_multiplier,
                sample_rate=sample_rate, truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size, count=num_steps
            )

        # Get advantage
        advantage = acc.get_advantage(state)
        return PrivacyMetrics(advantage=advantage)

    return evaluator


def find_noise_multiplier_for_advantage(
    advantage: float,
    num_steps: int,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
    advantage_tol: float = 1e-2,
    noise_min: float = 0.1,
    noise_max: float = 50.0,
) -> float:
    """Calibrate noise_multiplier to achieve target advantage (f-DP).

    Args:
        advantage: Target advantage value in [0, 1].
        num_steps: Number of DP-SGD steps.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.
        advantage_tol: Convergence tolerance for advantage.
        noise_min: Lower bound for search.
        noise_max: Upper bound for search.

    Returns:
        Calibrated noise_multiplier.
    """
    evaluator = create_dpsgd_advantage_evaluator(
        num_steps=num_steps,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        batch_size=batch_size,
        dataset_size=dataset_size,
        truncated_batch_size=truncated_batch_size,
        grid_step=grid_step,
    )

    target = CalibrationTarget(kind="advantage", advantage=advantage)

    config = CalibrationConfig(
        param_min=noise_min,
        param_max=noise_max,
        target_tol=advantage_tol,
        increasing=False,  # Higher noise → lower advantage
    )

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    return result.parameter_value


def get_advantage_for_dpsgd(
    noise_multiplier: float,
    num_steps: int,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> float:
    """Compute advantage for DP-SGD with given parameters.

    Args:
        noise_multiplier: Noise scale parameter.
        num_steps: Number of DP-SGD steps.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.

    Returns:
        Advantage value.
    """
    evaluator = create_dpsgd_advantage_evaluator(
        num_steps=num_steps,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        batch_size=batch_size,
        dataset_size=dataset_size,
        truncated_batch_size=truncated_batch_size,
        grid_step=grid_step,
    )
    metrics = evaluator(noise_multiplier)
    return metrics.advantage


def create_dpsgd_beta_evaluator(
    num_steps: int,
    target_alpha: float,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> PrivacyEvaluator:
    """Create beta evaluator for DP-SGD (error rates).

    Args:
        num_steps: Number of DP-SGD steps.
        target_alpha: False positive rate for beta computation.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.

    Returns:
        PrivacyEvaluator that maps noise_multiplier → PrivacyMetrics.
    """
    # Parameter validation
    if sampling_method == "poisson":
        if sample_rate is None:
            raise ValueError("sample_rate required for poisson sampling")
    elif sampling_method == "fixed_batch":
        if batch_size is None or dataset_size is None:
            raise ValueError("batch_size and dataset_size required for fixed_batch")
    elif sampling_method == "truncated_poisson":
        if sample_rate is None or dataset_size is None or truncated_batch_size is None:
            raise ValueError(
                "sample_rate, dataset_size, and truncated_batch_size "
                "required for truncated_poisson"
            )

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate beta for given noise."""
        state = acc.create(discretization_interval=grid_step)

        # Compose based on sampling method
        if sampling_method == "poisson":
            state = acc.compose_poisson_gaussian(
                state, noise_multiplier=noise_multiplier,
                sample_rate=sample_rate, count=num_steps
            )
        elif sampling_method == "fixed_batch":
            state = acc.compose_sampled_gaussian(
                state, noise_multiplier=noise_multiplier,
                batch_size=batch_size, dataset_size=dataset_size, count=num_steps
            )
        elif sampling_method == "truncated_poisson":
            state = acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=noise_multiplier,
                sample_rate=sample_rate, truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size, count=num_steps
            )

        # Get beta at target alpha
        beta = acc.get_beta(state, alpha=target_alpha)
        return PrivacyMetrics(alpha=target_alpha, beta=beta)

    return evaluator


def find_noise_multiplier_for_err_rates(
    alpha: float,
    beta: float,
    num_steps: int,
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
    beta_tol: float = 1e-2,
    noise_min: float = 0.1,
    noise_max: float = 50.0,
) -> float:
    """Calibrate noise_multiplier to achieve target (α, β) error rates.

    Args:
        alpha: Target false positive rate in [0, 1].
        beta: Target false negative rate in [0, 1].
        num_steps: Number of DP-SGD steps.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.
        beta_tol: Convergence tolerance for beta.
        noise_min: Lower bound for search.
        noise_max: Upper bound for search.

    Returns:
        Calibrated noise_multiplier.
    """
    evaluator = create_dpsgd_beta_evaluator(
        num_steps=num_steps,
        target_alpha=alpha,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        batch_size=batch_size,
        dataset_size=dataset_size,
        truncated_batch_size=truncated_batch_size,
        grid_step=grid_step,
    )

    target = CalibrationTarget(kind="err_rates", alpha=alpha, beta=beta)

    config = CalibrationConfig(
        param_min=noise_min,
        param_max=noise_max,
        target_tol=beta_tol,
        increasing=True,  # Higher noise → higher beta
    )

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    return result.parameter_value


def get_beta_for_dpsgd(
    noise_multiplier: float,
    num_steps: int,
    alpha: Union[float, np.ndarray],
    sampling_method: Literal["poisson", "truncated_poisson", "fixed_batch"] = "poisson",
    sample_rate: float | None = None,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    truncated_batch_size: int | None = None,
    grid_step: float = 1e-3,
) -> Union[float, np.ndarray]:
    """Compute beta (FNR) for DP-SGD at given alpha (FPR).

    Args:
        noise_multiplier: Noise scale parameter.
        num_steps: Number of DP-SGD steps.
        alpha: False positive rate(s) in [0, 1]. Can be scalar or array.
        sampling_method: Sampling method.
        sample_rate: Poisson sampling probability.
        batch_size: Batch size.
        dataset_size: Dataset size.
        truncated_batch_size: Max batch size.
        grid_step: PLD discretization interval.

    Returns:
        False negative rate(s) corresponding to input alpha.
    """
    # Create state based on sampling method
    state = acc.create(discretization_interval=grid_step)

    # Parameter validation
    if sampling_method == "poisson":
        if sample_rate is None:
            raise ValueError("sample_rate required for poisson sampling")
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=noise_multiplier,
            sample_rate=sample_rate, count=num_steps
        )
    elif sampling_method == "fixed_batch":
        if batch_size is None or dataset_size is None:
            raise ValueError("batch_size and dataset_size required for fixed_batch")
        state = acc.compose_sampled_gaussian(
            state, noise_multiplier=noise_multiplier,
            batch_size=batch_size, dataset_size=dataset_size, count=num_steps
        )
    elif sampling_method == "truncated_poisson":
        if sample_rate is None or dataset_size is None or truncated_batch_size is None:
            raise ValueError(
                "sample_rate, dataset_size, and truncated_batch_size "
                "required for truncated_poisson"
            )
        state = acc.compose_truncated_poisson_gaussian(
            state, noise_multiplier=noise_multiplier,
            sample_rate=sample_rate, truncated_batch_size=truncated_batch_size,
            dataset_size=dataset_size, count=num_steps
        )

    return acc.get_beta(state, alpha=alpha)


__all__ = [
    # Core primitives (re-exported from riskcal)
    "PrivacyEvaluator",
    "PrivacyMetrics",
    "CalibrationTarget",
    "CalibrationConfig",
    "CalibrationResult",
    "calibrate_parameter",
    # Evaluator factories
    "create_dpsgd_epsilon_evaluator",
    "create_dpsgd_advantage_evaluator",
    "create_dpsgd_beta_evaluator",
    # Calibration functions
    "find_noise_multiplier_for_epsilon_delta",
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    # Query functions
    "get_epsilon_for_dpsgd",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
]
