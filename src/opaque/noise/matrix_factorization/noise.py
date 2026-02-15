"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The API follows the same ``(fn, state)`` pattern as ``gaussian_stateful``:
``matrix_factorization_noise`` returns ``(init_fn, noise_fn)`` where
``init_fn`` creates state from a gradient template and ``noise_fn``
applies correlated noise at each step.

Example:
    >>> from opaque.noise.matrix_factorization import identity, matrix_factorization_noise
    >>> init_fn, noise_fn = matrix_factorization_noise(identity(), stddev=1.0, seed=42)
    >>> state = init_fn(grad_template)
    >>> noisy_grad, state = noise_fn(clipped_grad, state)

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

from opaque.utils.pytree import tree_map

from . import streaming_matrix


@dataclasses.dataclass(frozen=True)
class MFNoiseState:
    """State for matrix factorization noise.

    Attributes:
        inner_state: Internal state (streaming matrix state or step counter).
        rng_state: Random number generator for reproducibility.
    """

    inner_state: Any
    rng_state: torch.Generator | None = None


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


def matrix_factorization_noise(
    noising: torch.Tensor | streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any], MFNoiseState],
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
]:
    """Create correlated noise functions using matrix factorization.

    This is the primary entry point for DP-FTRL-style noise addition. The
    ``noising`` argument represents C^{-1} in the factorization A = B @ C,
    where the noise covariance is proportional to (C^{-1})^T @ C^{-1}.

    Returns ``(init_fn, noise_fn)`` following the functional ``(fn, state)``
    pattern used by ``gaussian_noise_stateful``.

    Args:
        noising: Either a dense 2D tensor (``torch.Tensor``) or a
            ``StreamingMatrix`` representing C^{-1}.
        stddev: Standard deviation for the base noise.
        seed: Optional random seed for reproducibility.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(init_fn, noise_fn)`` where:
        - ``init_fn(grad_template) -> MFNoiseState``
        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``

    Example:
        >>> import torch
        >>> from opaque.noise.matrix_factorization.toeplitz import (
        ...     inverse_as_streaming_matrix,
        ...     optimal_max_error_strategy_coefs,
        ... )
        >>> coefs = optimal_max_error_strategy_coefs(100)
        >>> noising = inverse_as_streaming_matrix(coefs)
        >>> init_fn, noise_fn = matrix_factorization_noise(
        ...     noising, stddev=1.0, seed=42,
        ... )
        >>> state = init_fn(grad_template)
        >>> noisy_grad, state = noise_fn(clipped_grad, state)
    """
    if isinstance(noising, torch.Tensor):
        return _dense_mf_noise(noising, stddev=stddev, seed=seed, dtype=dtype)
    elif isinstance(noising, streaming_matrix.StreamingMatrix):
        return _streaming_mf_noise(noising, stddev=stddev, seed=seed, dtype=dtype)
    else:
        raise TypeError(f"Unsupported noising type: {type(noising)}")


def _dense_mf_noise(
    noising: torch.Tensor,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, Callable]:
    """Init/noise functions from a dense noising matrix C^{-1}."""
    if noising.ndim != 2:
        raise ValueError(f"Expected 2D matrix, found shape {noising.shape}")

    def init_fn(grad_template):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        return MFNoiseState(
            inner_state=torch.tensor(0, dtype=torch.long),
            rng_state=gen,
        )

    def noise_fn(clipped_grads, state):
        index = state.inner_state
        gen = state.rng_state
        max_steps = noising.shape[0]
        if index >= max_steps:
            raise ValueError(
                f"Step {index} exceeds noising matrix size {max_steps}. "
                f"The noising matrix must have at least as many rows as steps."
            )
        matrix_row = noising[index] * stddev

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
        new_state = MFNoiseState(
            inner_state=index + 1,
            rng_state=gen,
        )
        return noisy_grads, new_state

    return init_fn, noise_fn


def _streaming_mf_noise(
    noising: streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    seed: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, Callable]:
    """Init/noise functions from a streaming noising matrix C^{-1}."""

    def init_fn(grad_template):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        streaming_state = noising.init_multiply(grad_template)
        return MFNoiseState(
            inner_state=streaming_state,
            rng_state=gen,
        )

    def noise_fn(clipped_grads, state):
        gen = state.rng_state
        streaming_state = state.inner_state

        # Generate IID noise
        iid_noise = _iid_normal_noise(clipped_grads, stddev, generator=gen, dtype=dtype)
        # Apply streaming matrix to correlate noise
        corr_noise, new_streaming_state = noising.multiply_next(
            iid_noise, streaming_state
        )
        # Add correlated noise to gradients
        noisy_grads = tree_map(
            lambda g, n: (g + n).to(g.dtype),
            clipped_grads,
            corr_noise,
        )
        new_state = MFNoiseState(
            inner_state=new_streaming_state,
            rng_state=gen,
        )
        return noisy_grads, new_state

    return init_fn, noise_fn
