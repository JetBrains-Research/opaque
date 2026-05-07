"""Internal Kalman / DiSK implementation (random-walk filter per tensor element)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.denoising.types import DenoiserState
from opaque.utils.per_group import PerGroup
from opaque.utils.pytree import tree_map_with_path


def _param_key_from_path(path: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in path)


def _measurement_variance_from_stddev(
    noise_stddev: float | PerGroup,
) -> float | PerGroup:
    if isinstance(noise_stddev, PerGroup):
        return PerGroup(
            noise_stddev.groups,
            {k: v * v for k, v in noise_stddev.values.items()},
        )
    return float(noise_stddev) * float(noise_stddev)


def _scalar_r_for_path(
    path: tuple[Any, ...],
    measurement_var: float | PerGroup,
) -> float:
    if isinstance(measurement_var, PerGroup):
        return measurement_var.for_key(_param_key_from_path(path))
    return float(measurement_var)


def _leaf_kalman_step(
    noisy: torch.Tensor,
    estimate: torch.Tensor,
    error_var: torch.Tensor,
    *,
    r: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Kalman update (predict + measurement) on a single tensor leaf."""
    dtype = torch.promote_types(
        noisy.dtype,
        torch.promote_types(estimate.dtype, error_var.dtype),
    )
    dtype = torch.promote_types(dtype, torch.float32)
    noisy_f = noisy.to(dtype)
    estimate_f = estimate.to(dtype)
    error_var_f = error_var.to(dtype)
    r = r.to(device=noisy.device, dtype=dtype)
    q = q.to(device=noisy.device, dtype=dtype)

    prior_var = error_var_f + q
    denom = prior_var + r
    gain = prior_var / denom.clamp_min(torch.finfo(dtype).tiny)
    innovation = noisy_f - estimate_f
    new_estimate = estimate_f + gain * innovation
    new_error_var = (1.0 - gain) * prior_var
    return new_estimate.to(noisy.dtype), new_error_var


@dataclasses.dataclass(frozen=True)
class DiskDenoiserState(DenoiserState):
    """Immutable state for :func:`~opaque.denoising.disk_denoiser` (DiSK)."""

    _estimate: Any
    _error_var: Any
    _step_counter: int


def _validate_noise_stddev(noise_stddev: float | PerGroup) -> None:
    if isinstance(noise_stddev, PerGroup):
        for g, v in noise_stddev.values.items():
            if v <= 0:
                raise ValueError(
                    f"noise_stddev must be positive for all groups, got {v} for '{g}'"
                )
    elif noise_stddev <= 0:
        raise ValueError(f"noise_stddev must be positive, got {noise_stddev}")


def disk_denoiser(
    grad_template: Any,
    *,
    noise_stddev: float | PerGroup,
    process_stddev: float,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[..., tuple[Any, DiskDenoiserState]],
    DiskDenoiserState,
]:
    """Build a DiSK denoiser for a gradient-shaped PyTree.

    Uses a random-walk state model and Gaussian measurement noise at the same
    scale as the DP mechanism (``noise_stddev``).  Each tensor element is filtered
    independently.

    Args:
        grad_template: PyTree with the same structure as noisy gradients; leaves
            must be tensors (shapes and devices define filtering).
        noise_stddev: Same units as :func:`~opaque.noise.gaussian_noise` (σ), scalar
            or :class:`~opaque.utils.per_group.PerGroup` when noise scales per group.
        process_stddev: Process noise scale (same units as ``noise_stddev``); the
            filter uses process variance ``Q = process_stddev ** 2``.
        dtype: Optional dtype for internal Kalman math (defaults to float32 minimum).

    Returns:
        ``(denoise, state)`` where ``denoise(noisy, state, *, noise_stddev=None)``
        returns ``(filtered, new_state)``.  Pass ``noise_stddev`` per call to match
        adaptive clipping.

    Raises:
        ValueError: If ``noise_stddev`` or ``process_stddev`` are invalid.
    """
    if process_stddev < 0:
        raise ValueError(f"process_stddev must be non-negative, got {process_stddev}")
    _validate_noise_stddev(noise_stddev)

    process_q = float(process_stddev) * float(process_stddev)
    default_measurement_var = _measurement_variance_from_stddev(noise_stddev)

    def _compute_dtype(leaf: torch.Tensor) -> torch.dtype:
        if dtype is not None:
            return torch.promote_types(dtype, torch.float32)
        return torch.promote_types(leaf.dtype, torch.float32)

    def _init_leaf(
        path: tuple[Any, ...], leaf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = _compute_dtype(leaf)
        r_scalar = _scalar_r_for_path(path, default_measurement_var)
        r_t = torch.tensor(r_scalar, dtype=compute_dtype, device=leaf.device)
        q_t = torch.tensor(process_q, dtype=compute_dtype, device=leaf.device)
        z = torch.zeros_like(leaf, dtype=compute_dtype)
        init_var = r_t + q_t
        init_var = torch.full_like(z, float(init_var.item()))
        return z, init_var

    estimate = tree_map_with_path(
        lambda path, leaf: _init_leaf(path, leaf)[0], grad_template
    )
    error_var = tree_map_with_path(
        lambda path, leaf: _init_leaf(path, leaf)[1], grad_template
    )

    state0 = DiskDenoiserState(
        _estimate=estimate,
        _error_var=error_var,
        _step_counter=0,
    )

    def denoise(
        noisy: Any,
        st: DiskDenoiserState,
        *,
        noise_stddev: float | PerGroup | None = None,
    ) -> tuple[Any, DiskDenoiserState]:
        if noise_stddev is not None:
            _validate_noise_stddev(noise_stddev)
            r_effective = _measurement_variance_from_stddev(noise_stddev)
        else:
            r_effective = default_measurement_var

        def _step_leaf(
            path: tuple[Any, ...],
            n_leaf: torch.Tensor,
            e_leaf: torch.Tensor,
            v_leaf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            r_scalar = _scalar_r_for_path(path, r_effective)
            compute_dtype = _compute_dtype(n_leaf)
            r_t = torch.tensor(r_scalar, dtype=compute_dtype, device=n_leaf.device)
            q_t = torch.tensor(process_q, dtype=compute_dtype, device=n_leaf.device)
            return _leaf_kalman_step(n_leaf, e_leaf, v_leaf, r=r_t, q=q_t)

        def _step_by_path(
            path: tuple[Any, ...], n_leaf: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            def _get(subtree: Any, p: tuple[Any, ...]) -> torch.Tensor:
                cur: Any = subtree
                for k in p:
                    if isinstance(cur, dict):
                        cur = cur[k]
                    else:
                        cur = cur[int(k)]
                return cur

            e_leaf = _get(st._estimate, path)
            v_leaf = _get(st._error_var, path)
            return _step_leaf(path, n_leaf, e_leaf, v_leaf)

        pairs = tree_map_with_path(_step_by_path, noisy)

        def _is_est_var_pair(t: Any) -> bool:
            return (
                isinstance(t, tuple)
                and len(t) == 2
                and isinstance(t[0], torch.Tensor)
                and isinstance(t[1], torch.Tensor)
            )

        def _unzip_est(t: Any) -> Any:
            if _is_est_var_pair(t):
                return t[0]
            if isinstance(t, dict):
                return {k: _unzip_est(v) for k, v in t.items()}
            if isinstance(t, (list, tuple)):
                return type(t)(_unzip_est(x) for x in t)
            raise TypeError(f"unexpected node in Kalman unzip: {type(t)}")

        def _unzip_var(t: Any) -> Any:
            if _is_est_var_pair(t):
                return t[1]
            if isinstance(t, dict):
                return {k: _unzip_var(v) for k, v in t.items()}
            if isinstance(t, (list, tuple)):
                return type(t)(_unzip_var(x) for x in t)
            raise TypeError(f"unexpected node in Kalman unzip: {type(t)}")

        new_estimate = _unzip_est(pairs)
        new_error_var = _unzip_var(pairs)

        new_state = DiskDenoiserState(
            _estimate=new_estimate,
            _error_var=new_error_var,
            _step_counter=st._step_counter + 1,
        )
        return new_estimate, new_state

    return denoise, state0
