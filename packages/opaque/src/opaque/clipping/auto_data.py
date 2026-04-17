"""Data-dependent AUTO-S clipping (Auto DP-SGD Phase 3).

Extends AUTO-S with a data-dependent clipping threshold computed from the
batch's aggregate gradient norm.  A safety clip at ``safety_clip_norm``
provides the formal L2 sensitivity bound; the data-dependent threshold
``C_t = threshold_scale * ||mean_grad||`` adapts to each batch for better
utility.

Accounting must use ``acc.auto_clip_gaussian()`` (not ``acc.gaussian()``)
to correctly handle the noise variance change between neighboring datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import grad_and_value, vmap as _vmap

from opaque.clipping._helpers import (
    batch_size_from_args,
    normalize_fun_to_return_aux,
    normalize_to_tuple,
    zero_grads_like,
)
from opaque.clipping.clipped_fun import _sum_clipped_tensor
from opaque.clipping.pytree import clip_pytree, scale_pytree_auto_s
from opaque.clipping.types import ClipState
from opaque.utils.pytree import global_norm, tree_map

from opaque.clipping.auto import AutoClippedGradAux


@dataclass(frozen=True)
class DataDependentAutoClipState(ClipState):
    """State for data-dependent AUTO-S clipping.

    The formal sensitivity bound is ``safety_clip_norm / normalize_by``,
    same as standard fixed clipping.  The data-dependent threshold
    ``C_t`` adapts each step but is capped at ``safety_clip_norm``.

    Accounting must use ``acc.auto_clip_gaussian()`` because the noise
    std (calibrated to ``C_t``) changes between neighboring datasets.

    Attributes:
        clipping_norm: The safety clip norm (formal L2 bound).
        normalize_by: Divisor applied to the scaled gradient sum.
        gamma: Stability constant for AUTO-S (> 0).
        threshold_scale: Scale factor ``W`` for computing ``C_t = W * ||mean_grad||``.
        last_threshold: The data-dependent threshold from the most recent step.
    """

    clipping_norm: float
    normalize_by: float
    gamma: float
    threshold_scale: float
    last_threshold: float

    def __post_init__(self):
        if self.clipping_norm <= 0:
            raise ValueError(
                f"clipping_norm must be positive, got {self.clipping_norm}"
            )
        if self.normalize_by <= 0:
            raise ValueError(f"normalize_by must be positive, got {self.normalize_by}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.threshold_scale <= 0:
            raise ValueError(
                f"threshold_scale must be positive, got {self.threshold_scale}"
            )


def data_dependent_auto_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    safety_clip_norm: float,
    threshold_scale: float = 1.0,
    gamma: float = 0.01,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    return_aux: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, DataDependentAutoClipState]:
    """Compute AUTO-S-scaled gradients with a data-dependent threshold.

    Two-pass algorithm per batch:

    1. Compute per-example gradients, safety-clip each to ``safety_clip_norm``.
    2. Derive ``C_t = min(threshold_scale * ||mean(clipped_grads)||, safety_clip_norm)``.
    3. Scale each (safety-clipped) gradient by ``C_t / (||g_i|| + gamma)``.
    4. Sum and divide by ``normalize_by``.

    Noise should be calibrated to ``noise_multiplier * C_t / normalize_by``
    (the actual per-step sensitivity), and accounting must use
    ``acc.auto_clip_gaussian(sensitivity, noise_ratio, dimension)`` to
    handle the data-dependent noise variance.

    Args:
        loss_fn: Scalar loss function.
        argnums: Which argument(s) to differentiate w.r.t.
        has_aux: If True, ``loss_fn`` returns ``(value, loss_aux)``.
        safety_clip_norm: Hard L2 norm bound applied before threshold
            computation. Provides the formal sensitivity guarantee.
        threshold_scale: Scale factor ``W`` for ``C_t = W * ||mean_grad||``.
        gamma: Stability constant for AUTO-S (> 0, default 0.01).
        normalize_by: Divide scaled sum by this constant.
        batch_argnums: Which argument(s) have a batch dimension.
        return_aux: If True, return per-example diagnostics.
        pre_clipping_transform: Optional per-example gradient transform.
        dtype: Optional output dtype.

    Returns:
        ``(grad_fn, initial_state)`` where ``grad_fn`` has the same
        call signature as other clipping processes.
    """
    if safety_clip_norm <= 0:
        raise ValueError(f"safety_clip_norm must be positive, got {safety_clip_norm}")
    if threshold_scale <= 0:
        raise ValueError(f"threshold_scale must be positive, got {threshold_scale}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if normalize_by <= 0.0:
        raise ValueError(f"normalize_by must be > 0, got {normalize_by}.")

    argnums_tuple = normalize_to_tuple(argnums)
    batch_argnums_tuple = normalize_to_tuple(batch_argnums)

    if not batch_argnums_tuple:
        raise ValueError("Batch argnums must not be empty.")
    if min(argnums_tuple + batch_argnums_tuple) < 0:
        raise ValueError(
            f"argnums={argnums_tuple} and batch_argnums={batch_argnums_tuple} "
            f"must be >= 0."
        )
    shared = set(argnums_tuple) & set(batch_argnums_tuple)
    if shared:
        raise ValueError(
            "Cannot compute clipped gradients for argnums that have a batch "
            f"axis. {argnums_tuple=} and {batch_argnums_tuple=} with overlap "
            f"{list(shared)}."
        )

    loss_fn_normalized = normalize_fun_to_return_aux(loss_fn, has_aux)
    grad_and_value_fn = grad_and_value(
        loss_fn_normalized, argnums=argnums, has_aux=True
    )

    initial_state = DataDependentAutoClipState(
        clipping_norm=safety_clip_norm,
        normalize_by=normalize_by,
        gamma=gamma,
        threshold_scale=threshold_scale,
        last_threshold=safety_clip_norm,
    )

    def _empty_batch_response(args, state):
        grads = zero_grads_like(args, argnums_tuple)
        new_state = DataDependentAutoClipState(
            clipping_norm=state.clipping_norm,
            normalize_by=state.normalize_by,
            gamma=state.gamma,
            threshold_scale=state.threshold_scale,
            last_threshold=0.0,
        )
        if return_aux:
            empty = torch.empty(0)
            grad_aux = AutoClippedGradAux(
                loss_values=None,
                grad_norms=empty,
                clipped_grad_norms=empty,
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
            )
            return (grads, grad_aux), new_state
        return grads, new_state

    def grad_fn(*args, state, **kwargs):
        bs = batch_size_from_args(args, batch_argnums_tuple)
        if bs == 0:
            return _empty_batch_response(args, state)

        in_dims = tuple(
            0 if i in batch_argnums_tuple else None for i in range(len(args))
        )

        # --- Pass 1: compute per-example gradients + safety clip ---
        def per_example_grad_and_clip(*args_single):
            grad, value_and_aux = grad_and_value_fn(*args_single, **kwargs)
            grad = pre_clipping_transform(grad)
            clipped, clip_aux = clip_pytree(grad, clipping_norm=safety_clip_norm)
            if return_aux or has_aux:
                result_aux: dict[str, Any] = {"norms": clip_aux.norm.detach()}
                result_aux["values"] = value_and_aux[0]
                if has_aux:
                    result_aux["value_aux"] = value_and_aux[1]
                return clipped, result_aux
            return clipped, {"norms": clip_aux.norm.detach()}

        out_dims = (0, 0)
        vmapped_clip = _vmap(
            per_example_grad_and_clip,
            in_dims=in_dims,
            out_dims=out_dims,
            randomness="same",
        )
        clipped_grads, pass1_aux = vmapped_clip(*args)

        # Compute mean gradient norm for threshold
        mean_grad = tree_map(
            lambda x: torch.mean(x.float(), dim=0),
            clipped_grads,
        )
        mean_grad_norm = global_norm(mean_grad).item()
        c_t = min(threshold_scale * mean_grad_norm, safety_clip_norm)
        c_t = max(c_t, 1e-10)  # avoid zero threshold

        # --- Pass 2: re-scale with data-dependent threshold ---
        def per_example_rescale(clipped_grad_single):
            scaled, scale_aux = scale_pytree_auto_s(
                clipped_grad_single,
                clipping_norm=c_t,
                gamma=gamma,
            )
            return scaled, scale_aux.norm.detach()

        vmapped_rescale = _vmap(
            per_example_rescale,
            in_dims=(0,),
            out_dims=(0, 0),
            randomness="same",
        )
        scaled_grads, scaled_norms_unused = vmapped_rescale(clipped_grads)

        # Sum across batch
        result = tree_map(
            lambda x: _sum_clipped_tensor(x, dim=0, requested_dtype=dtype),
            scaled_grads,
        )

        if normalize_by != 1.0:
            result = tree_map(lambda x: x / normalize_by, result)

        new_state = DataDependentAutoClipState(
            clipping_norm=state.clipping_norm,
            normalize_by=state.normalize_by,
            gamma=state.gamma,
            threshold_scale=state.threshold_scale,
            last_threshold=c_t,
        )

        if not return_aux:
            return result, new_state

        orig_norms = pass1_aux.get("norms")
        n = orig_norms.numel() if isinstance(orig_norms, torch.Tensor) else 0

        # Compute scaled norms for diagnostics
        scaled_result_norms = torch.zeros(n)
        if n > 0:
            for i in range(n):
                single_grad = tree_map(lambda x: x[i], scaled_grads)
                scaled_result_norms[i] = global_norm(single_grad).item()

        if n > 0:
            effective_cn = safety_clip_norm
            num_exceeded = float((orig_norms > effective_cn).sum().item())
            rate = num_exceeded / max(1.0, float(n))
        else:
            rate = None

        grad_aux = AutoClippedGradAux(
            loss_values=pass1_aux.get("values"),
            grad_norms=orig_norms,
            clipped_grad_norms=scaled_result_norms,
            loss_aux=pass1_aux.get("value_aux"),
            clipping_rate=rate,
            batch_size=n,
        )
        return (result, grad_aux), new_state

    return grad_fn, initial_state


__all__ = [
    "data_dependent_auto_clipped_grad",
    "DataDependentAutoClipState",
]
