"""Gaussian noise generation for differential privacy."""

import torch

from opaque.utils.pytree import tree_map

"""Gaussian noise addition for differential privacy.

This module provides functions for adding calibrated Gaussian noise to gradients
for DP-SGD (Differentially Private Stochastic Gradient Descent).
"""


def add_gaussian_noise(
    grads,
    stddev: float,
    generator: torch.Generator | None = None,
):
    """Add Gaussian noise to gradients for differential privacy.

    Adds i.i.d. Gaussian noise N(0, stddev²) to each element of the gradient
    PyTree. The noise standard deviation should be calibrated as:
        stddev = noise_multiplier × sensitivity
    where sensitivity comes from the clipping bound.

    Args:
        grads: Gradient PyTree (tensor, dict, tuple, or nested structure of tensors)
        stddev: Standard deviation of Gaussian noise
        generator: Optional torch.Generator for reproducibility

    Returns:
        Noisy gradients with same structure and dtypes as input

    Raises:
        ValueError: If stddev is negative

    Example:
        >>> import torch
        >>> from opaque.clipping import clipped_grad
        >>> from opaque.noise import add_gaussian_noise
        >>>
        >>> # After clipping
        >>> clipped_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
        >>> per_ex_grads = clipped_fn(params, batch_data)
        >>>
        >>> # Aggregate
        >>> sum_grads = per_ex_grads.sum(dim=0)
        >>>
        >>> # Add noise
        >>> sensitivity = clipped_fn.sensitivity()
        >>> stddev = noise_multiplier * sensitivity
        >>> generator = torch.Generator().manual_seed(42)
        >>> noisy = add_gaussian_noise(sum_grads, stddev, generator)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if stddev == 0:
        # No noise (for testing/debugging)
        return grads

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


__all__ = ["add_gaussian_noise"]
