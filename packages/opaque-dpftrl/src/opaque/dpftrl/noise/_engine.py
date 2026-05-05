"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The user-facing entry point is ``mf_noise`` (in ``opaque.noise``).
Internally, strategy modules (``band_mf``, ``blt``, etc.)
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

from opaque.distributed import (
    is_distributed,
    register_sync_type,
    sync_object,
)
from opaque.core.noise import NOISE_STATE_FIELD_OPS, assert_rng_key_equal
from opaque.types import NoiseState
from opaque.random import RngKey, generator_from_key
from opaque.random import fold_in as rng_fold_in
from opaque.core.pytree import tree_map

from . import _streaming_matrix as streaming_matrix


@dataclasses.dataclass(frozen=True)
class MFNoiseState(NoiseState):
    """State for matrix factorization noise.

    Attributes:
        _inner_state: Internal state (streaming matrix state or step counter).
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
        _first_max_norm: ``ClippedPytree.max_norm`` from the first call, latched by
            the dispatcher to enforce constant per-step sensitivity.  ``None``
            until the first call.  MF privacy analyses assume the per-step
            sensitivity is constant across the sequence; varying it (e.g.
            via adaptive clipping) breaks the standard proof, so the
            dispatcher rejects subsequent calls whose ``max_norm`` differs.
    """

    _inner_state: Any
    _step_counter: int
    _rng_key: RngKey
    _first_max_norm: float | None = None


def _internal_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use at least float32 for internal MF noise computations."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _iid_normal_noise(
    target_tree: Any,
    stddev: float,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Generate IID normal noise matching the structure of target_tree."""

    def _randn_on_device(
        shape: tuple[int, ...],
        *,
        noise_dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        try:
            return torch.randn(
                shape, dtype=noise_dtype, device=device, generator=generator
            )
        except RuntimeError as exc:
            if "device type for generator" in str(exc):
                return torch.randn(shape, dtype=noise_dtype, generator=generator).to(
                    device=device
                )
            raise

    def make_noise(t):
        noise_dtype = _internal_compute_dtype(dtype or t.dtype)
        noise = _randn_on_device(
            t.shape,
            noise_dtype=noise_dtype,
            device=t.device,
            generator=generator,
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
        try:
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        except RuntimeError as exc:
            if "device type for generator" in str(exc):
                noise = torch.randn(shape, dtype=dtype, generator=generator).to(
                    device=device
                )
            else:
                raise
        result = result + coef * noise
    return result


def _matrix_factorization_noise(
    grad_template: Any,
    noising: torch.Tensor | streaming_matrix.StreamingMatrix,
    *,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Internal: create ``(noise_fn, state)`` from a noising matrix.

    This is the engine that all strategy modules call. Users
    should use ``mf_noise`` (or a strategy wrapper) instead.

    Args:
        grad_template: Pytree with the same structure/shapes as gradients.
        noising: Dense 2D tensor or ``StreamingMatrix`` representing C^{-1}.
        key: Pre-resolved base ``RngKey``.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        ``(noise_fn, state)`` where
        ``noise_fn(grads, state, *, stddev) -> (noised, state)``.  ``stddev``
        is the per-step standard deviation for the base IID noise; the
        dispatcher derives it from ``noise_multiplier * ClippedPytree.max_norm``.
    """
    if isinstance(noising, torch.Tensor):
        return _tensor_mf_noise(grad_template, noising, key=key, dtype=dtype)
    elif isinstance(noising, streaming_matrix.StreamingMatrix):
        return _streaming_mf_noise(grad_template, noising, key=key, dtype=dtype)
    else:
        raise TypeError(f"Unsupported noising type: {type(noising)}")


def _tensor_mf_noise(
    grad_template: Any,
    noising: torch.Tensor,
    *,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a 2D noising matrix C^{-1}."""
    if noising.ndim != 2:
        raise ValueError(f"Expected 2D matrix, found shape {noising.shape}")

    state = MFNoiseState(
        _inner_state=torch.tensor(0, dtype=torch.long),
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(clipped_grads, st, *, stddev: float):
        effective_stddev = float(stddev)
        index = st._inner_state
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)
        max_steps = noising.shape[0]
        if index >= max_steps:
            raise ValueError(
                f"Step {index} exceeds noising matrix size {max_steps}. "
                f"The noising matrix must have at least as many rows as steps."
            )
        matrix_row = noising[index] * effective_stddev

        def add_noise(grad_tensor):
            compute_dtype = _internal_compute_dtype(dtype or grad_tensor.dtype)
            noise = _gaussian_linear_combination(
                matrix_row,
                grad_tensor.shape,
                compute_dtype,
                grad_tensor.device,
                generator=g,
            )
            return (grad_tensor + noise).to(grad_tensor.dtype)

        noisy_grads = tree_map(add_noise, clipped_grads)
        new_state = MFNoiseState(
            _inner_state=index + 1,
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state


def _streaming_mf_noise(
    grad_template: Any,
    noising: streaming_matrix.StreamingMatrix,
    *,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a streaming noising matrix C^{-1}."""
    streaming_state = noising.init_multiply(grad_template)
    state = MFNoiseState(
        _inner_state=streaming_state,
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(clipped_grads, st, *, stddev: float):
        effective_stddev = float(stddev)
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)
        s_state = st._inner_state

        iid_noise = _iid_normal_noise(
            clipped_grads,
            effective_stddev,
            generator=g,
            dtype=dtype,
        )
        corr_noise, new_streaming_state = noising.multiply_next(iid_noise, s_state)
        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )
        new_state = MFNoiseState(
            _inner_state=new_streaming_state,
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state


# ---- Distributed state validation ----


_MF_NOISE_STATE_FIELD_OPS: dict[str, str] = {
    **NOISE_STATE_FIELD_OPS,
    "_first_max_norm": "assert_equal",
}


def sync_mf_noise_state(state: MFNoiseState) -> MFNoiseState:
    """Validate MF noise state consistency across ranks.

    Asserts that all ranks share the same seed, step counter, and (once
    latched) first-call sensitivity bound.  No-op outside
    ``torch.distributed``.  Registered automatically with
    :func:`opaque.distributed.sync`.
    """
    if not is_distributed():
        return state
    assert_rng_key_equal(state, "MFNoiseState")
    return sync_object(state, field_ops=_MF_NOISE_STATE_FIELD_OPS)


register_sync_type(MFNoiseState, sync_mf_noise_state)
