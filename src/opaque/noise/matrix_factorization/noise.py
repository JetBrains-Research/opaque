"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The user-facing entry point is ``custom_mf_noise`` (in ``opaque.noise``).
Internally, recipe wrappers (``band_mf_noise``, ``blt_mf_noise``, etc.)
call ``_matrix_factorization_noise`` which returns ``(noise_fn, state)``.

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


def _matrix_factorization_noise(
    grad_template: Any,
    noising: torch.Tensor | streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    gen: torch.Generator,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Internal: create ``(noise_fn, state)`` from a noising matrix.

    This is the engine that all ``*_mf_noise`` wrappers call. Users
    should use ``custom_mf_noise`` (or a recipe wrapper) instead.

    Args:
        grad_template: Pytree with the same structure/shapes as gradients.
        noising: Dense 2D tensor or ``StreamingMatrix`` representing C^{-1}.
        stddev: Standard deviation for the base noise.
        gen: Pre-resolved ``torch.Generator``.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        ``(noise_fn, state)`` where ``noise_fn(grads, state) -> (noisy, state)``.

    Note:
        In distributed settings, the generator uses the **same seed** across all
        devices (centralized pattern). This matches Opaque's standard DDP approach
        where noise is conceptually added after gradient aggregation.
        
        The ``gen`` parameter is expected to be pre-resolved via
        ``_resolve_generator()`` which handles distributed mode automatically.
    """
    if isinstance(noising, torch.Tensor):
        return _dense_mf_noise(
            grad_template, noising, stddev=stddev, gen=gen, dtype=dtype
        )
    elif isinstance(noising, streaming_matrix.StreamingMatrix):
        return _streaming_mf_noise(
            grad_template, noising, stddev=stddev, gen=gen, dtype=dtype
        )
    else:
        raise TypeError(f"Unsupported noising type: {type(noising)}")


def _dense_mf_noise(
    grad_template: Any,
    noising: torch.Tensor,
    *,
    stddev: float,
    gen: torch.Generator,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a dense noising matrix C^{-1}."""
    if noising.ndim != 2:
        raise ValueError(f"Expected 2D matrix, found shape {noising.shape}")

    state = MFNoiseState(
        inner_state=torch.tensor(0, dtype=torch.long),
        rng_state=gen,
    )

    def noise_fn(clipped_grads, st):
        index = st.inner_state
        g = st.rng_state
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
                generator=g,
            )
            return (grad_tensor + noise).to(grad_tensor.dtype)

        noisy_grads = tree_map(add_noise, clipped_grads)
        new_state = MFNoiseState(
            inner_state=index + 1,
            rng_state=g,
        )
        return noisy_grads, new_state

    return noise_fn, state


def _streaming_mf_noise(
    grad_template: Any,
    noising: streaming_matrix.StreamingMatrix,
    *,
    stddev: float,
    gen: torch.Generator,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a streaming noising matrix C^{-1}."""
    streaming_state = noising.init_multiply(grad_template)
    state = MFNoiseState(
        inner_state=streaming_state,
        rng_state=gen,
    )

    def noise_fn(clipped_grads, st):
        g = st.rng_state
        s_state = st.inner_state

        iid_noise = _iid_normal_noise(clipped_grads, stddev, generator=g, dtype=dtype)
        corr_noise, new_streaming_state = noising.multiply_next(iid_noise, s_state)
        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )
        new_state = MFNoiseState(
            inner_state=new_streaming_state,
            rng_state=g,
        )
        return noisy_grads, new_state

    return noise_fn, state
