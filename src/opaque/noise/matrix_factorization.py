"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The key abstraction is the ``matrix_factorization_privatizer`` which
wraps either a dense matrix or a StreamingMatrix and returns a stateful
noise-addition transform compatible with any PyTorch optimizer.

Example:
    >>> from opaque.matrix_factorization import streaming_matrix
    >>> from opaque.noise.matrix_factorization import (
    ...     matrix_factorization_privatizer,
    ... )
    >>> # Standard Gaussian noise (identity matrix = independent noise)
    >>> privatizer = gaussian_privatizer(stddev=1.0)
    >>> state = privatizer.init(model_params)
    >>> noisy_grad, state = privatizer.update(clipped_grad, state)

References:
    - Correlated noise mechanisms: https://arxiv.org/abs/2506.08201
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT mechanisms: https://arxiv.org/abs/2404.16706
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.matrix_factorization import streaming_matrix
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class PrivatizerState:
    """State for a noise-addition privatizer.

    Attributes:
        inner_state: Internal state (varies by implementation).
        rng_state: Optional random state for reproducibility.
    """

    inner_state: Any
    rng_state: torch.Generator | None = None


@dataclasses.dataclass(frozen=True)
class Privatizer:
    """A stateful noise-addition transform for DP gradient processing.

    This is the PyTorch equivalent of an optax.GradientTransformation.
    Privatizers add noise to clipped gradients and maintain state across
    training steps.

    Attributes:
        init: Initialize the privatizer state from model parameters.
        update: Apply noise to gradients and return (noisy_grad, new_state).
    """

    init: Callable[[Any], PrivatizerState]
    update: Callable[[Any, PrivatizerState], tuple[Any, PrivatizerState]]


def _iid_normal_noise(
    target_tree: Any,
    stddev: float,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Generate IID normal noise matching the structure of target_tree."""

    def make_noise(t):
        noise_dtype = dtype or t.dtype
        noise = torch.randn(
            t.shape, dtype=noise_dtype, device=t.device, generator=generator
        )
        return noise * stddev

    return tree_map(make_noise, target_tree)


def _gaussian_linear_combination(
    matrix_row: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute a linear combination of standard Gaussian random variables.

    Given coefficients [c0, c1, ..., c_{k-1}] and IID z_0, ..., z_{k-1}
    ~ N(0, I), returns sum_i c_i * z_i.
    """
    nonzero = matrix_row != 0
    first = int(nonzero.long().argmax().item()) if nonzero.any() else 0
    last = (
        len(matrix_row) - int(nonzero.flip(0).long().argmax().item())
        if nonzero.any()
        else 0
    )

    result = torch.zeros(shape, dtype=dtype, device=device)
    for idx in range(first, last):
        coef = matrix_row[idx].to(dtype)
        noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        result = result + coef * noise
    return result


def matrix_factorization_privatizer(
    noising_matrix: torch.Tensor | streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> Privatizer:
    """Create a privatizer that adds correlated noise using matrix factorization.

    This is the primary entry point for DP-FTRL-style noise addition. The
    ``noising_matrix`` represents C^{-1} in the factorization A = B @ C,
    where the noise covariance is proportional to (C^{-1})^T @ C^{-1}.

    Args:
        noising_matrix: Either a dense matrix (torch.Tensor) or a
            StreamingMatrix representing C^{-1}.
        stddev: Standard deviation for the base noise.
        seed: Optional random seed for reproducibility.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A Privatizer with init() and update() methods.

    Example:
        >>> import torch
        >>> from opaque.matrix_factorization.toeplitz import (
        ...     inverse_as_streaming_matrix,
        ...     optimal_max_error_strategy_coefs,
        ... )
        >>> coefs = optimal_max_error_strategy_coefs(100)
        >>> noising = inverse_as_streaming_matrix(coefs)
        >>> privatizer = matrix_factorization_privatizer(
        ...     noising, stddev=1.0
        ... )
    """
    if isinstance(noising_matrix, torch.Tensor):
        return _dense_matrix_factorization_privatizer(
            noising_matrix, stddev=stddev, seed=seed, dtype=dtype
        )
    elif isinstance(noising_matrix, streaming_matrix.StreamingMatrix):
        return _streaming_matrix_factorization_privatizer(
            noising_matrix, stddev=stddev, seed=seed, dtype=dtype
        )
    else:
        raise TypeError(f"Unsupported noising_matrix type: {type(noising_matrix)}")


def gaussian_privatizer(
    *, stddev: float, seed: int | None = None, dtype: torch.dtype | None = None
) -> Privatizer:
    """Create a privatizer that adds independent Gaussian noise.

    This is the standard DP-SGD noise addition mechanism (special case
    of matrix factorization with identity matrix).

    Args:
        stddev: Standard deviation of the Gaussian noise.
        seed: Optional random seed.
        dtype: Optional dtype for noise.

    Returns:
        A Privatizer that adds IID Gaussian noise at each step.
    """
    return matrix_factorization_privatizer(
        streaming_matrix.identity(),
        stddev=stddev,
        seed=seed,
        dtype=dtype,
    )


def _dense_matrix_factorization_privatizer(
    noising_matrix: torch.Tensor,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> Privatizer:
    """Privatizer from a dense noising matrix C^{-1}."""
    if noising_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, found shape {noising_matrix.shape}")

    def init(model_params):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        return PrivatizerState(
            inner_state=torch.tensor(0, dtype=torch.long),
            rng_state=gen,
        )

    def update(clipped_grads, state):
        index = state.inner_state
        gen = state.rng_state
        max_steps = noising_matrix.shape[0]
        if index >= max_steps:
            raise ValueError(
                f"Privatizer step {index} exceeds noising_matrix size {max_steps}. "
                f"The noising matrix must have at least as many rows as optimizer steps."
            )
        matrix_row = noising_matrix[index] * stddev

        def add_noise(grad_tensor):
            noise = _gaussian_linear_combination(
                matrix_row,
                grad_tensor.shape,
                dtype or grad_tensor.dtype,
                grad_tensor.device,
                generator=gen,
            )
            return (grad_tensor + noise).to(grad_tensor.dtype)

        noisy_grads = tree_map(add_noise, clipped_grads)
        new_state = PrivatizerState(
            inner_state=index + 1,
            rng_state=gen,
        )
        return noisy_grads, new_state

    return Privatizer(init=init, update=update)


def _streaming_matrix_factorization_privatizer(
    noising_matrix: streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> Privatizer:
    """Privatizer from a streaming noising matrix C^{-1}."""

    def init(model_params):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        # Initialize streaming state from model structure
        streaming_state = noising_matrix.init_multiply(model_params)
        return PrivatizerState(
            inner_state=streaming_state,
            rng_state=gen,
        )

    def update(clipped_grads, state):
        gen = state.rng_state
        streaming_state = state.inner_state

        # Generate IID noise
        iid_noise = _iid_normal_noise(clipped_grads, stddev, generator=gen, dtype=dtype)
        # Apply streaming matrix to correlate noise
        corr_noise, new_streaming_state = noising_matrix.multiply_next(
            iid_noise, streaming_state
        )
        # Add correlated noise to gradients
        noisy_grads = tree_map(
            lambda g, n: (g + n).to(g.dtype),
            clipped_grads,
            corr_noise,
        )
        new_state = PrivatizerState(
            inner_state=new_streaming_state,
            rng_state=gen,
        )
        return noisy_grads, new_state

    return Privatizer(init=init, update=update)
