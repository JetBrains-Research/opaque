"""AUTO-S automatic gradient clipping (Bu et al., NeurIPS 2023).

Replaces hard per-example clipping ``min(1, R/||g||)`` with smooth scaling
``R / (||g|| + gamma)``.  The L2 sensitivity is still bounded by ``R``, so
existing Gaussian noise calibration and PLD accounting apply unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import grad_and_value

from opaque.clipping._helpers import (
    batch_size_from_args,
    normalize_fun_to_return_aux,
    normalize_to_tuple,
    zero_grads_like,
)
from opaque.clipping.clipped_fun import _microbatch_accumulate, _sum_clipped_tensor
from opaque.clipping.pytree import scale_pytree_auto_s
from opaque.clipping.types import ClipState
from opaque.utils.per_group import PerGroup
from opaque.utils.pytree import global_norm, tree_map

from torch.func import vmap as _vmap


@dataclass(frozen=True)
class AutoClippedGradAux:
    """Diagnostic outputs from auto_clipped_grad.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use
    ``ClipState.sensitivity`` for noise calibration.

    Inherits the same field schema as :class:`ClippedGradAux`.
    """

    loss_values: Any | None = None
    grad_norms: Any | None = None
    clipped_grad_norms: Any | None = None
    loss_aux: Any | None = None
    clipping_rate: float | None = None
    batch_size: int = 0
    group_norms: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class AutoClipState(ClipState):
    """Immutable state for AUTO-S automatic gradient clipping.

    AUTO-S scales each per-example gradient by ``R / (||g|| + gamma)``
    instead of hard-clipping with ``min(1, R / ||g||)``.  The output norm
    is strictly below ``R`` for all finite inputs, so the L2 sensitivity
    remains ``R / normalize_by`` — identical to fixed clipping.

    Accounting is therefore unchanged: use ``acc.gaussian(nm)`` (no
    ``acc.adaclip`` wrapper needed).

    Attributes:
        clipping_norm: Reference norm ``R`` (the formal sensitivity bound).
        normalize_by: Divisor applied to the scaled gradient sum.
        gamma: Stability constant (> 0).  Controls behavior near zero-norm
            gradients.  Paper default is 0.01.
    """

    clipping_norm: float | PerGroup
    normalize_by: float
    gamma: float

    def __post_init__(self):
        if isinstance(self.clipping_norm, PerGroup):
            for gname, val in self.clipping_norm.values.items():
                if val <= 0:
                    raise ValueError(
                        f"clipping_norm must be positive for all groups, "
                        f"got {val} for group '{gname}'"
                    )
        else:
            if self.clipping_norm <= 0:
                raise ValueError(
                    f"clipping_norm must be positive, got {self.clipping_norm}"
                )
        if self.normalize_by <= 0:
            raise ValueError(f"normalize_by must be positive, got {self.normalize_by}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")


def auto_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    clipping_norm: float | PerGroup,
    gamma: float = 0.01,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    return_aux: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    microbatch_size: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, AutoClipState]:
    """Create a function computing the sum of AUTO-S-scaled gradients.

    AUTO-S (Bu et al., NeurIPS 2023) replaces the hard per-example clip
    ``min(1, R/||g||)`` with smooth scaling ``R / (||g|| + gamma)``.  The
    output norm for each example is strictly below ``R``, so L2 sensitivity
    is ``R / normalize_by`` — identical to ``clipped_grad`` — and all
    existing noise calibration and PLD accounting apply unchanged.

    Example::

        >>> import torch
        >>> from opaque.clipping import auto_clipped_grad
        >>> f = lambda param, data: 0.5 * ((data - param) ** 2).mean()
        >>> g, clip_state = auto_clipped_grad(f, clipping_norm=1.0, gamma=0.01)
        >>> result, clip_state = g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]), state=clip_state)

    Accounting:
        Use ``acc.gaussian(nm)`` with Poisson amplification — **no**
        ``acc.adaclip`` wrapper — because the sensitivity bound is the
        same fixed ``R / normalize_by`` as standard clipping.

    Args:
        loss_fn: Scalar loss function.  If ``has_aux``, returns
            ``(scalar, loss_aux)``.
        argnums: Which argument(s) to differentiate w.r.t.
        has_aux: If True, ``loss_fn`` returns ``(value, loss_aux)``.
        clipping_norm: Reference norm ``R``.  Formal L2 sensitivity bound.
        gamma: Stability constant (> 0, default 0.01).
        normalize_by: Divide scaled sum by this constant.
        batch_argnums: Which argument(s) contain the batch dimension.
        return_aux: If True, return per-example diagnostics alongside grads.
        pre_clipping_transform: Optional per-example gradient transform
            applied before scaling (same as in ``clipped_grad``).
        microbatch_size: Optional microbatch size for memory-efficient
            processing.
        dtype: Optional output dtype for the gradient sum.

    Returns:
        ``(auto_clipped_grad_fn, initial_state)`` where:
        - ``auto_clipped_grad_fn(*args, state=...) -> (grads, new_state)``
          or ``((grads, aux), new_state)`` when ``return_aux=True``.
        - ``initial_state``: :class:`AutoClipState`.
    """
    if isinstance(clipping_norm, PerGroup):
        for gname, val in clipping_norm.values.items():
            if val <= 0:
                raise ValueError(
                    f"clipping_norm must be positive for all groups, "
                    f"got {val} for group '{gname}'"
                )
    elif clipping_norm <= 0:
        raise ValueError(f"clipping_norm must be positive, got {clipping_norm}")
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
            f"argnums={argnums_tuple} and batch_argnums={batch_argnums_tuple} must be >= 0."
        )
    shared = set(argnums_tuple) & set(batch_argnums_tuple)
    if shared:
        raise ValueError(
            "Cannot compute clipped gradients for argnums that have a batch axis. "
            f"{argnums_tuple=} and {batch_argnums_tuple=} with overlap {list(shared)}."
        )

    loss_fn_normalized = normalize_fun_to_return_aux(loss_fn, has_aux)

    grad_and_value_fn = grad_and_value(
        loss_fn_normalized, argnums=argnums, has_aux=True
    )

    def per_example_grad_fn(*args, **kwargs):
        grad, value_and_aux = grad_and_value_fn(*args, **kwargs)
        result = pre_clipping_transform(grad)
        if return_aux:
            aux_dict: dict[str, Any] = {}
            aux_dict["values"] = value_and_aux[0]
            if has_aux:
                aux_dict["value_aux"] = value_and_aux[1]
            return result, aux_dict
        return result

    def per_example_scale_fn(*args_single):
        if return_aux:
            raw_grad, inner_aux = per_example_grad_fn(*args_single)
        else:
            raw_grad = per_example_grad_fn(*args_single)
            inner_aux = {}

        scaled_value, norm_aux = scale_pytree_auto_s(
            raw_grad,
            clipping_norm=clipping_norm,
            gamma=gamma,
        )

        if return_aux:
            aux_dict = {
                "norms": norm_aux.norm.detach(),
                "clipped_norms": global_norm(scaled_value).detach(),
            }
            if norm_aux.group_norms is not None:
                aux_dict["group_norms"] = {
                    k: v.detach() for k, v in norm_aux.group_norms.items()
                }
            if isinstance(inner_aux, dict):
                if "values" in inner_aux:
                    val = inner_aux["values"]
                    aux_dict["values"] = (
                        val.detach() if isinstance(val, torch.Tensor) else val
                    )
                if "value_aux" in inner_aux:
                    aux_dict["value_aux"] = inner_aux["value_aux"]
            return scaled_value, aux_dict
        return scaled_value

    initial_state = AutoClipState(
        clipping_norm=clipping_norm,
        normalize_by=normalize_by,
        gamma=gamma,
    )

    def _empty_batch_response(args, state):
        grads = zero_grads_like(args, argnums_tuple)
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
            return (grads, grad_aux), state
        return grads, state

    def auto_clipped_grad_fn(*args, state, **kwargs):
        if batch_size_from_args(args, batch_argnums_tuple) == 0:
            return _empty_batch_response(args, state)

        in_dims = tuple(
            0 if i in batch_argnums_tuple else None for i in range(len(args))
        )

        if microbatch_size is None:
            out_dims = 0 if not return_aux else (0, 0)
            vmapped = _vmap(
                per_example_scale_fn,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness="same",
            )
            if return_aux:
                scaled_values, aux = vmapped(*args)
            else:
                scaled_values = vmapped(*args)
                aux = ()

            result = tree_map(
                lambda x: _sum_clipped_tensor(x, dim=0, requested_dtype=dtype),
                scaled_values,
            )
        else:
            result, aux = _microbatch_accumulate(
                per_example_fn=per_example_scale_fn,
                args=args,
                batch_argnums=batch_argnums_tuple,
                in_dims=in_dims,
                microbatch_size=microbatch_size,
                return_aux=return_aux,
                dtype=dtype,
            )

        if normalize_by != 1.0:
            result = tree_map(lambda x: x / normalize_by, result)

        if not return_aux:
            return result, state

        aux_dict = aux if isinstance(aux, dict) else {}
        norms = aux_dict.get("norms")
        group_norms_dict = aux_dict.get("group_norms")
        bs = norms.numel() if isinstance(norms, torch.Tensor) else 0

        if isinstance(norms, torch.Tensor) and bs > 0:
            if isinstance(clipping_norm, PerGroup) and group_norms_dict is not None:
                any_exceeded = torch.zeros(bs, dtype=torch.bool, device=norms.device)
                for gname, gnorms in group_norms_dict.items():
                    any_exceeded = any_exceeded | (gnorms > clipping_norm.values[gname])
                num_exceeded = float(any_exceeded.sum().item())
            else:
                effective_cn = (
                    clipping_norm.effective
                    if isinstance(clipping_norm, PerGroup)
                    else clipping_norm
                )
                num_exceeded = float((norms > effective_cn).sum().item())
            rate = num_exceeded / max(1.0, float(bs))
        else:
            rate = None

        grad_aux = AutoClippedGradAux(
            loss_values=aux_dict.get("values"),
            grad_norms=norms,
            clipped_grad_norms=aux_dict.get("clipped_norms"),
            loss_aux=aux_dict.get("value_aux"),
            clipping_rate=rate,
            batch_size=bs,
            group_norms=group_norms_dict,
        )
        return (result, grad_aux), state

    return auto_clipped_grad_fn, initial_state


__all__ = ["auto_clipped_grad", "AutoClipState", "AutoClippedGradAux"]
