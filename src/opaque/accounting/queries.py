"""Query functions for extracting privacy parameters from accounting state.

This module provides thin wrappers around dp_accounting's PLD query methods,
plus integration with riskcal for alpha/beta (operational risk) queries.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import riskcal

from dp_accounting.pld import privacy_loss_distribution as pld_lib

# Type alias for clarity
PrivacyState = pld_lib.PrivacyLossDistribution


def get_epsilon(state: PrivacyState, delta: float) -> float:
    """Compute epsilon for a given delta.

    Returns the smallest ε such that the mechanism satisfies (ε, δ)-differential
    privacy.

    Args:
        state: The privacy state (PLD) to query.
        delta: The target delta value. Must be in (0, 1).

    Returns:
        The epsilon value for (ε, δ)-DP.

    Raises:
        ValueError: If delta is not in (0, 1).

    Examples:
        >>> state = create()
        >>> state = compose_poisson_gaussian(
        ...     state, noise_multiplier=1.0, sample_rate=0.01, count=100
        ... )
        >>> epsilon = get_epsilon(state, delta=1e-5)
    """
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    return float(state.get_epsilon_for_delta(delta))


def get_delta(state: PrivacyState, epsilon: float) -> float:
    """Compute delta for a given epsilon.

    Returns the smallest δ such that the mechanism satisfies (ε, δ)-differential
    privacy for the given ε.

    Args:
        state: The privacy state (PLD) to query.
        epsilon: The target epsilon value. Must be non-negative.

    Returns:
        The delta value for (ε, δ)-DP.

    Raises:
        ValueError: If epsilon is negative.

    Examples:
        >>> state = create()
        >>> state = compose_poisson_gaussian(
        ...     state, noise_multiplier=1.0, sample_rate=0.01, count=100
        ... )
        >>> delta = get_delta(state, epsilon=1.0)
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    return float(state.get_delta_for_epsilon(epsilon))


def get_beta(
    state: PrivacyState,
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute FNR (beta) for given FPR (alpha) using f-DP tradeoff curves.

    This uses riskcal to compute the operational privacy risk: the false
    negative rate (FNR/beta) of a membership inference attack given a
    false positive rate (FPR/alpha).

    Args:
        state: The privacy state (PLD) to query.
        alpha: False positive rate(s). Can be a scalar or array in [0, 1].

    Returns:
        False negative rate(s) corresponding to the given FPR(s).
        Returns same type as input (scalar or array).

    Raises:
        ValueError: If any alpha value is not in [0, 1].

    Examples:
        Single FPR:
        >>> beta = get_beta(state, alpha=0.01)  # FNR at 1% FPR

        Multiple FPRs:
        >>> alphas = np.array([0.001, 0.01, 0.05, 0.1])
        >>> betas = get_beta(state, alpha=alphas)
    """
    # Validate alpha values
    alpha_array = np.atleast_1d(alpha)
    if np.any((alpha_array < 0) | (alpha_array > 1)):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Convert PLD to PLRV and compute beta using riskcal
    plrv = riskcal.conversions.plrvs_from_pld(state)
    return riskcal.plrv.get_beta(plrv, alpha=alpha)


def get_advantage(state: PrivacyState) -> float:
    """Compute maximum attack advantage (f-DP advantage).

    The advantage is the maximum true positive rate (TPR) achievable by
    any membership inference attack. Equivalently, it's the delta at epsilon=0.

    Args:
        state: The privacy state (PLD) to query.

    Returns:
        The maximum attack advantage in [0, 1].

    Examples:
        >>> state = create()
        >>> state = compose_poisson_gaussian(
        ...     state, noise_multiplier=1.0, sample_rate=0.01, count=100
        ... )
        >>> advantage = get_advantage(state)
        >>> print(f"Maximum attack advantage: {advantage:.4f}")
    """
    advantage = float(riskcal.conversions.get_advantage_from_pld(state))
    # Clamp to [0, 1] to handle numerical precision issues
    return max(0.0, min(1.0, advantage))


def get_privacy_curve(
    state: PrivacyState,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute complete f-DP tradeoff curve (FPR vs FNR).

    Returns the full tradeoff curve mapping false positive rates to
    false negative rates, providing a complete characterization of
    privacy-utility tradeoff for membership inference attacks.

    Args:
        state: The privacy state (PLD) to query.
        alphas: Array of FPR values at which to evaluate the curve.
            Must be in [0, 1] and sorted.

    Returns:
        Tuple of (alphas, betas) where:
        - alphas: The input FPR values (echoed back)
        - betas: Corresponding FNR values

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> state = create()
        >>> state = compose_poisson_gaussian(
        ...     state, noise_multiplier=1.0, sample_rate=0.01, count=100
        ... )
        >>> alphas = np.linspace(0, 1, 100)
        >>> _, betas = get_privacy_curve(state, alphas)
        >>> plt.plot(alphas, betas)
        >>> plt.xlabel("FPR (α)")
        >>> plt.ylabel("FNR (β)")
        >>> plt.title("f-DP Tradeoff Curve")
    """
    betas = get_beta(state, alpha=alphas)
    return alphas, betas
