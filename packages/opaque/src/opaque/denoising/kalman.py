"""Scalar Kalman filtering on gradient-shaped PyTrees (DiSK-style denoising).

Applies an independent random-walk Kalman filter per tensor element.  This is
post-processing on noisy gradients: it does not change the DP mechanism or
privacy budget when applied only to the released noisy signal.

``noise_var`` (measurement variance R) may be a scalar or :class:`~opaque.utils.per_group.PerGroup`
when noise scales differ per parameter group (e.g. MSE-optimal per-group noise).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.utils.per_group import PerGroup
from opaque.utils.pytree import tree_map_with_path


def _param_key_from_path(path: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in path)


def _scalar_r_for_path(
    path: tuple[Any, ...],
    noise_var: float | PerGroup,
) -> float:
    if isinstance(noise_var, PerGroup):
        return noise_var.for_key(_param_key_from_path(path))
    return float(noise_var)


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
class KalmanDenoiserState:
    """Immutable state for :func:`kalman_denoiser`."""

    _estimate: Any
    _error_var: Any
    _step_counter: int


# Alias for API stability until multiple denoiser state types exist.
DenoiserState = KalmanDenoiserState


def kalman_denoiser(
    grad_template: Any,
    *,
    noise_var: float | PerGroup,
    process_var: float,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[..., tuple[Any, KalmanDenoiserState]],
    KalmanDenoiserState,
]:
    """Build a Kalman denoiser for a gradient-shaped PyTree.

    Uses a random-walk state model (process variance ``process_var``) and
    Gaussian measurement noise with variance ``noise_var`` (R).  Each tensor
    element is filtered independently.

    Args:
        grad_template: PyTree with the same structure as noisy gradients; leaves
            must be tensors (shapes and devices define filtering).
        noise_var: Measurement variance R (scalar) or :class:`~opaque.utils.per_group.PerGroup`
            of variances keyed like the gradient dict (for per-group noise).
        process_var: Process noise variance Q (scalar random walk per step).
        dtype: Optional dtype for internal Kalman math (defaults to float32 minimum).

    Returns:
        ``(denoise, state)`` where ``denoise(noisy, state, *, noise_var=None)``
        returns ``(filtered, new_state)``.  Pass ``noise_var`` per call to match
        adaptive clipping (R is public: it is the mechanism noise variance).

    Raises:
        ValueError: If ``noise_var`` or ``process_var`` are not valid variances.
    """
    if process_var < 0:
        raise ValueError(f"process_var must be non-negative, got {process_var}")
    if isinstance(noise_var, PerGroup):
        for g, v in noise_var.values.items():
            if v <= 0:
                raise ValueError(
                    f"noise_var must be positive for all groups, got {v} for '{g}'"
                )
    elif noise_var <= 0:
        raise ValueError(f"noise_var must be positive, got {noise_var}")

    default_noise_var = noise_var

    def _compute_dtype(leaf: torch.Tensor) -> torch.dtype:
        if dtype is not None:
            return torch.promote_types(dtype, torch.float32)
        return torch.promote_types(leaf.dtype, torch.float32)

    def _init_leaf(
        path: tuple[Any, ...], leaf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = _compute_dtype(leaf)
        r_scalar = _scalar_r_for_path(path, default_noise_var)
        r_t = torch.tensor(r_scalar, dtype=compute_dtype, device=leaf.device)
        q_t = torch.tensor(process_var, dtype=compute_dtype, device=leaf.device)
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

    state0 = KalmanDenoiserState(
        _estimate=estimate,
        _error_var=error_var,
        _step_counter=0,
    )

    def denoise(
        noisy: Any,
        st: KalmanDenoiserState,
        *,
        noise_var: float | PerGroup | None = None,
    ) -> tuple[Any, KalmanDenoiserState]:
        r_effective = noise_var if noise_var is not None else default_noise_var
        if isinstance(r_effective, PerGroup):
            for g, v in r_effective.values.items():
                if v <= 0:
                    raise ValueError(
                        f"noise_var must be positive for all groups, got {v} for '{g}'"
                    )
        elif r_effective <= 0:
            raise ValueError(f"noise_var must be positive, got {r_effective}")

        def _step_leaf(
            path: tuple[Any, ...],
            n_leaf: torch.Tensor,
            e_leaf: torch.Tensor,
            v_leaf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            r_scalar = _scalar_r_for_path(path, r_effective)
            compute_dtype = _compute_dtype(n_leaf)
            r_t = torch.tensor(r_scalar, dtype=compute_dtype, device=n_leaf.device)
            q_t = torch.tensor(process_var, dtype=compute_dtype, device=n_leaf.device)
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

        new_state = KalmanDenoiserState(
            _estimate=new_estimate,
            _error_var=new_error_var,
            _step_counter=st._step_counter + 1,
        )
        return new_estimate, new_state

    return denoise, state0
