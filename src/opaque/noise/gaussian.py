"""Gaussian noise generation for differential privacy.

This module provides higher-order functions for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The functional API provides:
1. `gaussian(stddev)` - Returns a stateless noise function (recommended)
2. `gaussian_stateful(stddev, seed)` - Returns (fn, state) for reproducibility
"""

from collections.abc import Callable

import torch

from opaque.utils.pytree import tree_map


def gaussian(stddev: float) -> Callable:
    """Create a stateless Gaussian noise function.

    Returns a function that adds calibrated Gaussian noise N(0, stddev²) to gradients.
    This is the recommended API for typical use cases where reproducibility is not critical.

    For reproducible noise (e.g., testing, debugging), use `gaussian_stateful()`.

    Args:
        stddev: Standard deviation of Gaussian noise (usually `noise_multiplier * clip_norm`)

    Returns:
        A function `noise_fn(grads) -> noisy_grads` that adds calibrated noise

    Example:
        >>> import torch
        >>> from opaque.clipping import clipped_grad
        >>> from opaque.noise import gaussian
        >>>
        >>> # Configure gradient clipping
        >>> grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
        >>>
        >>> # Configure noise (user does multiplication)
        >>> noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        >>>     grads = grad_fn(params, batch['x'], batch['y'])
        >>>     noisy_grads = noise_fn(grads)  # Natural composition
        >>>     params = optimizer.step(params, noisy_grads)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if stddev == 0:
        # No noise (for testing/debugging)
        return lambda grads: grads

    def noise_fn(grads):
        """Add Gaussian noise to gradients."""

        def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
            """Add noise to a single tensor, preserving dtype and device."""
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return tensor + noise * stddev

        return tree_map(add_noise_to_tensor, grads)

    return noise_fn


def gaussian_stateful(stddev: float, seed: int) -> tuple[Callable, torch.Generator]:
    """Create a Gaussian noise function with explicit state management.

    Returns a tuple (noise_fn, state) where state is a torch.Generator for
    reproducible noise. This follows the functional pattern of explicit state passing.

    Use this when you need reproducible noise (e.g., for testing, debugging,
    or deterministic training). For typical use cases, use `gaussian()`.

    Args:
        stddev: Standard deviation of Gaussian noise
        seed: Random seed for the PRNG state

    Returns:
        A tuple (noise_fn, state) where:
        - noise_fn(grads, state) -> noisy_grads
        - state is a torch.Generator initialized with seed

    Example:
        >>> import torch
        >>> from opaque.noise import gaussian_stateful
        >>>
        >>> # Create noise function with explicit state
        >>> noise_fn, state = gaussian_stateful(stddev=1.1, seed=42)
        >>>
        >>> # Use in training (pass state explicitly)
        >>> for batch in dataloader:
        >>>     grads = compute_gradients(params, batch)
        >>>     noisy_grads = noise_fn(grads, state)  # Reproducible noise
        >>>     params = optimizer.step(params, noisy_grads)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    # Create generator with seed
    state = torch.Generator().manual_seed(seed)

    if stddev == 0:
        # No noise (for testing/debugging)
        return lambda grads, gen: grads, state

    def noise_fn(grads, generator: torch.Generator):
        """Add Gaussian noise to gradients using the provided generator."""

        def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
            """Add noise to a single tensor, preserving dtype and device."""
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                generator=generator,
            )
            return tensor + noise * stddev

        return tree_map(add_noise_to_tensor, grads)

    return noise_fn, state


__all__ = ["gaussian", "gaussian_stateful"]
