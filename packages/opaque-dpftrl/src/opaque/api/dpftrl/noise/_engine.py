"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The user-facing entry point is ``mf_gaussian_noise`` (in ``opaque.noise``).
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

from opaque.types import NoiseState, PerGroup
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in
from opaque.pytree import tree_map

from . import _streaming_matrix as streaming_matrix


@dataclasses.dataclass(frozen=True)
class MFNoiseState(NoiseState):
    """State for matrix factorization noise.

    Attributes:
        _inner_state: Internal state (streaming matrix state or step counter).
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
        _first_max_norm: ``ClippedPytree.max_norm`` from the first call (scalar
            or :class:`~opaque.types.PerGroup`), latched by the dispatcher to
            enforce constant per-step sensitivity.  ``None`` until the first
            call.  MF privacy analyses assume the per-step sensitivity is
            constant across the sequence.  Both fixed clipping
            (:func:`opaque.dpftrl.clipping.clipped_grad`) and AUTO-S clipping
            (:func:`opaque.dpftrl.clipping.auto_clipped_grad`) satisfy this
            assumption — the per-record bound is fixed at construction and
            does not depend on data.  Adaptive clipping
            (:func:`opaque.dpsgd.clipping.adaptive_clipped_grad`) breaks it,
            so the dispatcher rejects subsequent calls whose ``max_norm``
            differs.
        _first_max_norm_sync_fingerprint: Deterministic 64-bit fingerprint
            (computed at latch time) for cheap cross-rank checks in
            :func:`sync_mf_noise_state`, for both scalar and ``PerGroup`` norms.
            ``None`` before the first call.
    """

    _inner_state: Any
    _step_counter: int
    _rng_key: RngKey
    _first_max_norm: float | PerGroup | None = None
    _first_max_norm_sync_fingerprint: int | None = None


def _internal_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use at least float32 for internal MF noise computations."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _iid_normal_noise(
    target_tree: Any,
    stddev: float | PerGroup,
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

    if isinstance(stddev, PerGroup):
        if not isinstance(target_tree, dict):
            raise TypeError(
                "PerGroup stddev for MF noise requires a flat dict of tensors "
                "(matching make_functional trainable params), got "
                f"{type(target_tree).__name__}."
            )
        out: dict[str, Any] = {}
        for param_key, tensor in target_tree.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "PerGroup MF noise expects dict[str, Tensor] leaves; "
                    f"got {type(tensor).__name__} for key {param_key!r}."
                )
            leaf_std = stddev.for_key(param_key)
            noise_dtype = _internal_compute_dtype(dtype or tensor.dtype)
            noise = _randn_on_device(
                tensor.shape,
                noise_dtype=noise_dtype,
                device=tensor.device,
                generator=generator,
            )
            out[param_key] = noise * leaf_std
        return out

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
    should use ``mf_gaussian_noise`` (or a strategy wrapper) instead.

    Args:
        grad_template: Pytree with the same structure/shapes as gradients.
        noising: Dense 2D tensor or ``StreamingMatrix`` representing C^{-1}.
        key: Pre-resolved base ``RngKey``.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        ``(noise_fn, state)`` where
        ``noise_fn(grads, state, *, stddev) -> (noised, state)``.  ``stddev``
        is the per-step standard deviation for the base IID noise (scalar or
        :class:`~opaque.types.PerGroup` for per-parameter-group allocation);
        the dispatcher derives it from ``noise_multiplier`` and
        ``ClippedPytree.max_norm``.
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

    def noise_fn(clipped_grads, st, *, stddev: float | PerGroup):
        index = st._inner_state
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)
        max_steps = noising.shape[0]
        if index >= max_steps:
            raise ValueError(
                f"Step {index} exceeds noising matrix size {max_steps}. "
                f"The noising matrix must have at least as many rows as steps."
            )
        matrix_row_base = noising[index]

        if isinstance(stddev, PerGroup):
            if not isinstance(clipped_grads, dict):
                raise TypeError(
                    "PerGroup stddev for dense MF noise requires a flat dict of "
                    "tensors (matching make_functional trainable params), got "
                    f"{type(clipped_grads).__name__}."
                )

            def add_noise_keyed(param_key: str, grad_tensor: torch.Tensor):
                eff = stddev.for_key(param_key)
                matrix_row = matrix_row_base * eff
                compute_dtype = _internal_compute_dtype(dtype or grad_tensor.dtype)
                noise = _gaussian_linear_combination(
                    matrix_row,
                    grad_tensor.shape,
                    compute_dtype,
                    grad_tensor.device,
                    generator=g,
                )
                return (grad_tensor + noise).to(grad_tensor.dtype)

            noisy_grads = {}
            for k, v in clipped_grads.items():
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        "PerGroup dense MF noise expects dict[str, Tensor] leaves; "
                        f"got {type(v).__name__} for key {k!r}."
                    )
                noisy_grads[k] = add_noise_keyed(k, v)
        else:
            matrix_row = matrix_row_base * float(stddev)

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

    def noise_fn(clipped_grads, st, *, stddev: float | PerGroup):
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)
        s_state = st._inner_state

        iid_noise = _iid_normal_noise(
            clipped_grads,
            stddev,
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


# ---- Input validation helpers shared by mf_gaussian_noise + second-moment ----


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def _expect_clipped(value: Any, *, op: str):
    """Reject NoisedPytree or non-ClippedPytree inputs to the noise function."""
    from opaque.types import ClippedPytree, NoisedPytree

    if isinstance(value, NoisedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs, not NoisedPytree values that "
            "have already passed through a noise mechanism."
        )
    if not isinstance(value, ClippedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs. Wrap manual values with "
            "opaque.types.clipped(...)."
        )
    return value


def _validate_constant_max_norm(
    grads,
    first_max_norm: float | PerGroup | None,
    *,
    op: str,
) -> float | PerGroup:
    """Latch the per-step max_norm and reject changes across calls.

    MF privacy analyses calibrate noise from a sensitivity that is constant
    across the sequence.  Fixed clipping
    (:func:`opaque.dpftrl.clipping.clipped_grad`) and AUTO-S clipping
    (:func:`opaque.dpftrl.clipping.auto_clipped_grad`) both produce a
    constant ``ClippedPytree.max_norm`` and pass this latch.  Adaptive
    clipping (:func:`opaque.dpsgd.clipping.adaptive_clipped_grad`) varies
    its threshold across steps, which breaks the proof; the factory
    latches the first-call max_norm in the state and rejects any
    subsequent call whose max_norm differs.
    """
    max_norm = grads.max_norm
    if isinstance(max_norm, PerGroup):
        for group_name, value in max_norm.values.items():
            if value < 0:
                raise ValueError(
                    f"ClippedPytree max_norm must be non-negative for all groups, "
                    f"got {value} for group '{group_name}'."
                )
    else:
        if float(max_norm) < 0:
            raise ValueError(
                f"ClippedPytree max_norm must be non-negative, got {grads.max_norm}"
            )
    if first_max_norm is not None and max_norm != first_max_norm:
        raise ValueError(
            f"{op} saw a varying ClippedPytree.max_norm across calls "
            f"(first={first_max_norm}, now={max_norm}). MF privacy proofs "
            "assume a constant per-step sensitivity; this is satisfied by "
            "fixed and AUTO-S clipping but not by adaptive clipping, which "
            "is therefore unsupported with MF noise."
        )
    return max_norm


# ---- Distributed state validation ----
