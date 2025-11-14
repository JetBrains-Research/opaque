"""Composition functions for functional privacy accounting.

This module provides thin wrappers around Google's dp_accounting library,
using PrivacyLossDistribution (PLD) directly as the state. All functions
are pure: they take PLD as input and return a new PLD without mutation.

The PLD object is already functional and handles all the complex math
for privacy accounting. We just provide ergonomic wrappers.
"""

from __future__ import annotations

from dp_accounting.pld import privacy_loss_distribution as pld_lib
from dp_accounting.privacy_accountant import NeighboringRelation

# Type alias for clarity - PLD is our state
PrivacyState = pld_lib.PrivacyLossDistribution


def create(discretization_interval: float = 1e-4) -> PrivacyState:
    """Create identity privacy state (zero privacy cost).

    Args:
        discretization_interval: Discretization resolution for PLD.
            Smaller values give more accurate estimates but increase
            memory and computation. Default: 1e-4.

    Returns:
        Identity PLD representing zero privacy cost.

    Examples:
        >>> state = create()
        >>> epsilon = state.get_epsilon_for_delta(1e-5)
        >>> epsilon
        0.0
    """
    return pld_lib.identity(value_discretization_interval=discretization_interval)


def compose_poisson_gaussian(
    state: PrivacyState,
    noise_multiplier: float,
    sample_rate: float,
    count: int = 1,
    tail_mass_truncation: float = 1e-15,
) -> PrivacyState:
    """Compose Poisson-sampled Gaussian mechanism (standard DP-SGD).

    Args:
        state: Current privacy state (PLD).
        noise_multiplier: Noise multiplier (σ/sensitivity).
        sample_rate: Probability of sampling each example (batch_size / dataset_size).
        count: Number of training steps to compose.
        tail_mass_truncation: Probability mass to truncate from tails.

    Returns:
        New privacy state after composition.

    Examples:
        >>> state = create_identity()
        >>> state = compose_poisson_gaussian(
        ...     state, noise_multiplier=1.1, sample_rate=0.01, count=1000
        ... )
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    # Get discretization from current state
    discretization = state._pmf_remove._discretization  # pylint: disable=protected-access

    # Create Poisson-sampled Gaussian PLD
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        value_discretization_interval=discretization,
    )

    # Self-compose if count > 1
    if count > 1:
        pld = pld.self_compose(count, tail_mass_truncation)

    # Compose with current state
    return state.compose(pld, tail_mass_truncation)


def compose_sampled_gaussian(
    state: PrivacyState,
    noise_multiplier: float,
    batch_size: int,
    dataset_size: int,
    count: int = 1,
    tail_mass_truncation: float = 1e-15,
) -> PrivacyState:
    """Compose Gaussian mechanism with fixed-size batch sampling.

    This uses the Poisson approximation with doubled sensitivity for
    fixed-size sampling without replacement.

    Args:
        state: Current privacy state (PLD).
        noise_multiplier: Noise multiplier (σ/sensitivity).
        batch_size: Number of examples per batch (fixed).
        dataset_size: Total number of examples in dataset.
        count: Number of training steps to compose.
        tail_mass_truncation: Probability mass to truncate from tails.

    Returns:
        New privacy state after composition.

    Examples:
        >>> state = create_identity()
        >>> state = compose_sampled_gaussian(
        ...     state,
        ...     noise_multiplier=1.1,
        ...     batch_size=32,
        ...     dataset_size=1000,
        ...     count=1000
        ... )
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if dataset_size < batch_size:
        raise ValueError(
            f"dataset_size must be >= batch_size, got {dataset_size} < {batch_size}"
        )
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    # Convert to Poisson sampling with doubled sensitivity
    sample_rate = batch_size / dataset_size
    effective_noise_multiplier = noise_multiplier / 2.0

    return compose_poisson_gaussian(
        state=state,
        noise_multiplier=effective_noise_multiplier,
        sample_rate=sample_rate,
        count=count,
        tail_mass_truncation=tail_mass_truncation,
    )


def compose_truncated_poisson_gaussian(
    state: PrivacyState,
    noise_multiplier: float,
    sample_rate: float,
    truncated_batch_size: int,
    dataset_size: int,
    count: int = 1,
    tail_mass_truncation: float = 1e-15,
    neighboring_relation: NeighboringRelation = NeighboringRelation.ADD_OR_REMOVE_ONE,
) -> PrivacyState:
    """Compose truncated Poisson-sampled Gaussian mechanism.

    This provides tighter privacy bounds than standard Poisson sampling
    by bounding the batch size. See https://arxiv.org/abs/2508.15089

    Note: In dp-accounting 0.5.1, this may give looser bounds than expected.

    Args:
        state: Current privacy state (PLD).
        noise_multiplier: Noise multiplier (σ/sensitivity).
        sample_rate: Probability of sampling each example.
        truncated_batch_size: Maximum batch size.
        dataset_size: Total number of examples in dataset.
        count: Number of training steps to compose.
        tail_mass_truncation: Probability mass to truncate from tails.
        neighboring_relation: The neighboring relation for DP guarantee.

    Returns:
        New privacy state after composition.

    Examples:
        >>> state = create_identity()
        >>> state = compose_truncated_poisson_gaussian(
        ...     state,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     truncated_batch_size=100,
        ...     dataset_size=10000,
        ...     count=1000
        ... )
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if truncated_batch_size < 1:
        raise ValueError(
            f"truncated_batch_size must be at least 1, got {truncated_batch_size}"
        )
    if dataset_size < truncated_batch_size:
        raise ValueError(
            f"dataset_size must be >= truncated_batch_size, "
            f"got {dataset_size} < {truncated_batch_size}"
        )
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    # Get discretization from current state
    discretization = state._pmf_remove._discretization  # pylint: disable=protected-access

    # Create truncated Poisson PLD
    pld = pld_lib.from_truncated_subsampled_gaussian_mechanism(
        dataset_size=dataset_size,
        sampling_probability=sample_rate,
        truncated_batch_size=truncated_batch_size,
        noise_multiplier=noise_multiplier,
        value_discretization_interval=discretization,
        neighboring_relation=neighboring_relation,
    )

    # Self-compose if count > 1
    if count > 1:
        pld = pld.self_compose(count, tail_mass_truncation)

    # Compose with current state
    return state.compose(pld, tail_mass_truncation)


def compose_fixed_batch(
    state: PrivacyState,
    noise_multiplier: float,
    batch_size: int,
    dataset_size: int,
    count: int = 1,
    tail_mass_truncation: float = 1e-15,
) -> PrivacyState:
    """Compose Gaussian mechanism with fixed-size batch sampling (without replacement).

    This is an alias for compose_sampled_gaussian with clearer naming.
    Uses Poisson approximation with doubled sensitivity for fixed-size sampling.

    Note: For truly fixed-size batches, this uses a conservative approximation.
    Consider using truncated Poisson if you need tighter bounds.

    Args:
        state: Current privacy state (PLD).
        noise_multiplier: Noise multiplier (σ/sensitivity).
        batch_size: Number of examples per batch (fixed).
        dataset_size: Total number of examples in dataset.
        count: Number of training steps to compose.
        tail_mass_truncation: Probability mass to truncate from tails.

    Returns:
        New privacy state after composition.

    Examples:
        >>> state = create_identity()
        >>> # Train for 1000 steps with batch_size=32, dataset_size=1000
        >>> state = compose_fixed_batch(
        ...     state,
        ...     noise_multiplier=1.1,
        ...     batch_size=32,
        ...     dataset_size=1000,
        ...     count=1000
        ... )
    """
    return compose_sampled_gaussian(
        state=state,
        noise_multiplier=noise_multiplier,
        batch_size=batch_size,
        dataset_size=dataset_size,
        count=count,
        tail_mass_truncation=tail_mass_truncation,
    )
