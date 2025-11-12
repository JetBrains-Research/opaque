"""Calibration functions for DP hyperparameters.

This module provides functions to calibrate noise multipliers, batch sizes, and
number of training steps to achieve target (ε, δ)-DP guarantees.

Adapted from JAX-Privacy's calibration utilities.
"""

import math
from collections.abc import Callable

import scipy.optimize


def _solve_calibration(
    fn: Callable[[float], float], x_min: float, x_max: float, tol: float
) -> float:
    """Find x in [x_min, x_max] such that fn(x) is close to and at most 0.

    Uses Brent's method (scipy.optimize.brentq) for robust root finding.

    Args:
        fn: Function to find root of (we want fn(x) <= 0).
        x_min: Lower bound of search interval.
        x_max: Upper bound of search interval.
        tol: Tolerance for the solution.

    Returns:
        x such that fn(x) <= 0 and fn(x) is close to 0.
    """
    root, result = scipy.optimize.brentq(fn, x_min, x_max, xtol=tol, full_output=True)
    assert result.converged, "Root finding did not converge"

    # brentq guarantees a value in [root - tol, root + tol] such that fn(x) <= 0.
    # This value is not necessarily root, so we check root and endpoints.
    if fn(root) <= 0:
        return root
    elif fn(root - tol) <= 0:
        return root - tol
    elif fn(root + tol) <= 0:
        return root + tol
    else:
        # Slower but guaranteed to give x such that fn(x) <= 0
        return scipy.optimize.bisect(fn, x_min, x_max, xtol=tol)


def _resolve_accountant(accountant_type: str | type | Callable):
    """Resolve accountant_type to a callable that returns an accountant.

    Args:
        accountant_type: Either:
            - A string "pld" or "rdp"
            - A class (e.g., PLDAccountant)
            - A callable that returns an accountant instance

    Returns:
        A callable that returns an accountant instance.

    Raises:
        ValueError: If accountant_type is invalid.
    """
    # If it's a string, resolve to class
    if isinstance(accountant_type, str):
        if accountant_type == "pld":
            from opaque.accounting import PLDAccountant

            return PLDAccountant
        elif accountant_type == "rdp":
            from opaque.accounting import RDPAccountant

            return RDPAccountant
        else:
            raise ValueError(
                f"Unknown accountant_type string: {accountant_type}. Expected 'pld' or 'rdp'."
            )

    # If it's already a callable (class or function), return it
    if callable(accountant_type):
        return accountant_type

    raise ValueError(
        f"accountant_type must be a string ('pld', 'rdp'), a class, or a callable. "
        f"Got: {type(accountant_type)}"
    )


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    num_steps: int,
    accountant_type: str | type | Callable = "rdp",
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
    initial_max_noise: float = 100.0,
    initial_min_noise: float = 0.1,
    tol: float = 0.01,
) -> float:
    """Find noise multiplier that achieves target (ε, δ) privacy.

    Uses binary search to find the minimum noise_multiplier that satisfies
    the privacy budget after num_steps of training.

    Args:
        target_epsilon: Desired epsilon.
        target_delta: Desired delta.
        sample_rate: Sampling rate (batch_size / dataset_size).
        num_steps: Total number of training steps.
        accountant_type: Accountant to use. Can be:
            - String: "pld" or "rdp" (default: "rdp")
            - Class: e.g., PLDAccountant or RDPAccountant
            - Callable: lambda that returns an accountant instance
        truncated_batch_size: If using truncated Poisson (PLD only).
        dataset_size: Dataset size (required if using truncated Poisson).
        initial_max_noise: Initial upper bound for search.
        initial_min_noise: Initial lower bound for search.
        tol: Tolerance for noise multiplier.

    Returns:
        Calibrated noise_multiplier.

    Raises:
        ValueError: If no valid noise multiplier exists or parameters invalid.

    Example:
        >>> # Find noise for ε=3, δ=1e-5 over 1000 steps
        >>> noise_mult = calibrate_noise_multiplier(
        ...     target_epsilon=3.0,
        ...     target_delta=1e-5,
        ...     sample_rate=0.01,
        ...     num_steps=1000,
        ... )
        >>> print(f"Use noise_multiplier={noise_mult:.2f}")

        >>> # Use with custom accountant
        >>> from opaque.accounting import PLDAccountant
        >>> noise_mult = calibrate_noise_multiplier(
        ...     target_epsilon=3.0,
        ...     target_delta=1e-5,
        ...     sample_rate=0.01,
        ...     num_steps=1000,
        ...     accountant_type=PLDAccountant,
        ... )
    """
    accountant_cls = _resolve_accountant(accountant_type)

    # Validate truncated Poisson parameters
    if truncated_batch_size is not None:
        # Check if accountant supports truncated Poisson
        test_acc = accountant_cls()
        if not hasattr(test_acc, "step_truncated_poisson"):
            raise ValueError(
                "truncated_batch_size only supported with accountants that have "
                "step_truncated_poisson method (e.g., PLD accountant)"
            )
        if dataset_size is None:
            raise ValueError("dataset_size required when using truncated_batch_size")

    def get_epsilon(noise_multiplier: float) -> float:
        """Compute epsilon for given noise multiplier."""
        acc = accountant_cls()

        if truncated_batch_size is not None:
            acc.step_truncated_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size,
                num_steps=num_steps,
            )
        else:
            acc.step_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                num_steps=num_steps,
            )

        return acc.get_epsilon(target_delta=target_delta)

    # Expand search range if needed
    max_noise = initial_max_noise
    min_noise = initial_min_noise

    # Ensure max_noise gives epsilon <= target
    while get_epsilon(max_noise) > target_epsilon:
        min_noise, max_noise = max_noise, 2 * max_noise
        if max_noise > 1000:
            raise ValueError(
                f"Could not find noise multiplier for target_epsilon={target_epsilon}. "
                "Try increasing target_epsilon or decreasing num_steps."
            )

    # Find the minimum noise that achieves target epsilon
    def error_fn(nm: float) -> float:
        return get_epsilon(nm) - target_epsilon

    noise_multiplier = float(_solve_calibration(error_fn, min_noise, max_noise, tol))

    return noise_multiplier


def calibrate_steps(
    *,
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    sample_rate: float,
    accountant_type: str | type | Callable = "rdp",
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
    initial_max_steps: int = 10000,
    initial_min_steps: int = 1,
    tol: float = 1.0,
) -> int:
    """Find maximum number of steps that achieves target (ε, δ) privacy.

    Uses binary search to find the maximum number of training steps that
    stays within the privacy budget.

    Args:
        target_epsilon: Desired epsilon.
        target_delta: Desired delta.
        noise_multiplier: Noise multiplier to use.
        sample_rate: Sampling rate (batch_size / dataset_size).
        accountant_type: Accountant to use. Can be:
            - String: "pld" or "rdp" (default: "rdp")
            - Class: e.g., PLDAccountant or RDPAccountant
            - Callable: lambda that returns an accountant instance
        truncated_batch_size: If using truncated Poisson (PLD only).
        dataset_size: Dataset size (required if using truncated Poisson).
        initial_max_steps: Initial upper bound for search.
        initial_min_steps: Initial lower bound for search.
        tol: Tolerance for number of steps.

    Returns:
        Maximum number of training steps.

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> # Find max steps for ε=3, δ=1e-5
        >>> max_steps = calibrate_steps(
        ...     target_epsilon=3.0,
        ...     target_delta=1e-5,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ... )
        >>> print(f"Can train for {max_steps} steps")
    """
    accountant_cls = _resolve_accountant(accountant_type)

    # Validate truncated Poisson parameters
    if truncated_batch_size is not None:
        # Check if accountant supports truncated Poisson
        test_acc = accountant_cls()
        if not hasattr(test_acc, "step_truncated_poisson"):
            raise ValueError(
                "truncated_batch_size only supported with accountants that have "
                "step_truncated_poisson method (e.g., PLD accountant)"
            )
        if dataset_size is None:
            raise ValueError("dataset_size required when using truncated_batch_size")

    def get_epsilon(num_steps: int) -> float:
        """Compute epsilon for given number of steps."""
        acc = accountant_cls()

        if truncated_batch_size is not None:
            acc.step_truncated_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size,
                num_steps=num_steps,
            )
        else:
            acc.step_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                num_steps=num_steps,
            )

        return acc.get_epsilon(target_delta=target_delta)

    # Check if even 1 step exceeds budget
    if get_epsilon(initial_min_steps) > target_epsilon:
        raise ValueError(
            f"Even {initial_min_steps} step(s) exceeds target_epsilon={target_epsilon}. "
            "Try increasing noise_multiplier or target_epsilon."
        )

    # Expand search range if needed
    max_steps = initial_max_steps
    min_steps = initial_min_steps

    while get_epsilon(max_steps) < target_epsilon:
        min_steps, max_steps = max_steps, 2 * max_steps
        if max_steps > 1_000_000:
            raise ValueError(
                f"Could not find step limit for target_epsilon={target_epsilon}. "
                "Privacy budget is very large."
            )

    # Find maximum steps that stay within budget
    def error_fn(s: float) -> float:
        return get_epsilon(int(s)) - target_epsilon

    steps = int(math.floor(_solve_calibration(error_fn, min_steps, max_steps, tol)))

    return steps


def calibrate_batch_size(
    *,
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    num_steps: int,
    dataset_size: int,
    accountant_type: str | type | Callable = "rdp",
    truncated_batch_size: int | None = None,
    initial_max_batch_size: int | None = None,
    initial_min_batch_size: int = 1,
    tol: float = 1.0,
) -> int:
    """Find maximum batch size that achieves target (ε, δ) privacy.

    Uses binary search to find the maximum batch size (via sample_rate) that
    stays within the privacy budget.

    Args:
        target_epsilon: Desired epsilon.
        target_delta: Desired delta.
        noise_multiplier: Noise multiplier to use.
        num_steps: Total number of training steps.
        dataset_size: Total number of examples in dataset.
        accountant_type: Accountant to use. Can be:
            - String: "pld" or "rdp" (default: "rdp")
            - Class: e.g., PLDAccountant or RDPAccountant
            - Callable: lambda that returns an accountant instance
        truncated_batch_size: If using truncated Poisson (PLD only).
            If provided, this becomes the upper bound for calibration.
        initial_max_batch_size: Initial upper bound for search.
            Defaults to dataset_size.
        initial_min_batch_size: Initial lower bound for search.
        tol: Tolerance for batch size.

    Returns:
        Maximum batch size.

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> # Find max batch size for ε=3, δ=1e-5
        >>> max_batch_size = calibrate_batch_size(
        ...     target_epsilon=3.0,
        ...     target_delta=1e-5,
        ...     noise_multiplier=1.1,
        ...     num_steps=1000,
        ...     dataset_size=10000,
        ... )
        >>> print(f"Can use batch_size={max_batch_size}")
    """
    accountant_cls = _resolve_accountant(accountant_type)

    # Validate truncated Poisson parameters
    if truncated_batch_size is not None:
        # Check if accountant supports truncated Poisson
        test_acc = accountant_cls()
        if not hasattr(test_acc, "step_truncated_poisson"):
            raise ValueError(
                "truncated_batch_size only supported with accountants that have "
                "step_truncated_poisson method (e.g., PLD accountant)"
            )

    # Set default max batch size
    if initial_max_batch_size is None:
        initial_max_batch_size = dataset_size

    def get_epsilon(batch_size: int) -> float:
        """Compute epsilon for given batch size."""
        sample_rate = batch_size / dataset_size
        acc = accountant_cls()

        if truncated_batch_size is not None:
            acc.step_truncated_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                truncated_batch_size=truncated_batch_size,
                dataset_size=dataset_size,
                num_steps=num_steps,
            )
        else:
            acc.step_poisson(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                num_steps=num_steps,
            )

        return acc.get_epsilon(target_delta=target_delta)

    # Check if even smallest batch exceeds budget
    if get_epsilon(initial_min_batch_size) > target_epsilon:
        raise ValueError(
            f"Even batch_size={initial_min_batch_size} exceeds target_epsilon={target_epsilon}. "
            "Try increasing noise_multiplier or target_epsilon."
        )

    # Expand search range if needed (though unlikely for batch size)
    max_batch = initial_max_batch_size
    min_batch = initial_min_batch_size

    # Clamp to dataset size
    if max_batch > dataset_size:
        max_batch = dataset_size

    # Check if max batch size is within budget
    if get_epsilon(max_batch) <= target_epsilon:
        # Can use full dataset as batch
        return max_batch

    # Find maximum batch size that stays within budget
    def error_fn(b: float) -> float:
        return get_epsilon(int(b)) - target_epsilon

    batch_size = int(math.floor(_solve_calibration(error_fn, min_batch, max_batch, tol)))

    return batch_size
