"""Learning rate adjustment based on gradient clipping rate.

This module implements dynamic learning rate scaling that responds to the
empirical clipping rate, as described in DP-Adam-AC (Algorithm 1, lines 24-28).
"""


def clip_rate_based_lr_adjustment(
    current_lr_multiplier: float,
    clip_rate: float,
    target_clip_rate: float,
    clip_rate_low: float,
    clip_rate_high: float,
    increase_factor: float = 1.01,
    decrease_factor: float = 0.995,
    lr_multiplier_min: float = 0.1,
    lr_multiplier_max: float = 2.0,
) -> float:
    """Adjust learning rate multiplier based on observed clipping rate.

    From DP-Adam-AC Algorithm 1:
        if ρ < ρ_low:
            γ ← min(γ_max, γ · ↑)    # Increase LR (not clipping enough)
        else if ρ > ρ_high:
            γ ← max(γ_min, γ · ↓)    # Decrease LR (clipping too much)

    The intuition:
        - Low clip rate (ρ < ρ_low): Gradients are small, can increase LR
        - High clip rate (ρ > ρ_high): Gradients are large, should decrease LR
        - In target range [ρ_low, ρ_high]: Keep LR stable

    Args:
        current_lr_multiplier: Current learning rate multiplier (γ)
        clip_rate: Observed clipping rate (ρ), fraction in [0, 1]
        target_clip_rate: Target clipping rate (ρ*), typically 0.20
        clip_rate_low: Lower bound for acceptable clip rate (ρ_low)
        clip_rate_high: Upper bound for acceptable clip rate (ρ_high)
        increase_factor: Multiplicative increase when ρ < ρ_low (↑)
            Default: 1.01 (1% increase per adjustment)
        decrease_factor: Multiplicative decrease when ρ > ρ_high (↓)
            Default: 0.995 (0.5% decrease per adjustment)
        lr_multiplier_min: Minimum allowed multiplier (γ_min)
        lr_multiplier_max: Maximum allowed multiplier (γ_max)

    Returns:
        New learning rate multiplier γ, clamped to [γ_min, γ_max]

    Example:
        >>> # Start with neutral multiplier
        >>> gamma = 1.0
        >>>
        >>> # Observed clip rate is low (only 5% clipped)
        >>> gamma = clip_rate_based_lr_adjustment(
        ...     current_lr_multiplier=gamma,
        ...     clip_rate=0.05,
        ...     target_clip_rate=0.20,
        ...     clip_rate_low=0.10,
        ...     clip_rate_high=0.30,
        ... )
        >>> print(f"γ = {gamma:.4f}")  # Increased to ~1.01
        >>>
        >>> # Clip rate is too high (40% clipped)
        >>> gamma = clip_rate_based_lr_adjustment(
        ...     current_lr_multiplier=gamma,
        ...     clip_rate=0.40,
        ...     target_clip_rate=0.20,
        ...     clip_rate_low=0.10,
        ...     clip_rate_high=0.30,
        ... )
        >>> print(f"γ = {gamma:.4f}")  # Decreased to ~1.005
    """
    new_multiplier = current_lr_multiplier

    # Check if clip rate is outside acceptable range
    if clip_rate < clip_rate_low:
        # Too few gradients being clipped → increase LR
        new_multiplier = current_lr_multiplier * increase_factor
    elif clip_rate > clip_rate_high:
        # Too many gradients being clipped → decrease LR
        new_multiplier = current_lr_multiplier * decrease_factor
    # else: clip rate in acceptable range, keep LR stable

    # Clamp to valid range
    new_multiplier = max(lr_multiplier_min, min(lr_multiplier_max, new_multiplier))

    return float(new_multiplier)


def compute_clip_rate_thresholds(
    target_clip_rate: float,
    tolerance: float = 0.10,
) -> tuple[float, float]:
    """Compute clip rate operating range [ρ_low, ρ_high] from target.

    Provides reasonable defaults for the clip rate thresholds based on
    the target clip rate with symmetric tolerance.

    Args:
        target_clip_rate: Target clipping rate (ρ*)
        tolerance: Tolerance around target (default: 0.10 = ±10%)

    Returns:
        Tuple of (ρ_low, ρ_high) defining the acceptable range

    Example:
        >>> # For target 20% clipping, allow 10-30% range
        >>> low, high = compute_clip_rate_thresholds(0.20, tolerance=0.10)
        >>> print(f"ρ ∈ [{low:.2f}, {high:.2f}]")
        ρ ∈ [0.10, 0.30]
    """
    rho_low = max(0.0, target_clip_rate - tolerance)
    rho_high = min(1.0, target_clip_rate + tolerance)
    return rho_low, rho_high


__all__ = [
    "clip_rate_based_lr_adjustment",
    "compute_clip_rate_thresholds",
]
