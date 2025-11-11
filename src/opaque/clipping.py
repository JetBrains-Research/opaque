"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping per-example outputs (typically gradients)
to bounded L2 norms, which is a core primitive for DP-SGD training.

References:
    JAX-Privacy implementation:
    https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/src/experimental
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.func import grad as _torch_grad
from torch.func import vmap as _vmap

from opaque.pytree_utils import global_norm, tree_map


@dataclass(frozen=True)
class BoundedSensitivityCallable:
    """Callable with a sensitivity property.

    The function may return multiple outputs, some of which may have a batch
    axis and some of which may not. The sensitivity guarantee holds for all
    outputs that do not have a batch axis (i.e., because there was an aggregation
    over it). The auxiliary outputs with a batch axis are usually computed
    essentially for free so they are returned here, but must be handled with care
    by the caller (with respect to the DP guarantees, should they be needed).
    """

    fun: Callable[..., Any]
    l2_norm_bound: float

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
    nan_safe: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    Args:
        pytree: Dictionary of tensors to clip
        clip_norm: Maximum L2 norm (non-negative, or inf for no clipping)
        rescale_to_unit_norm: If True, additionally scale by 1/clip_norm
            so final norm is at most 1.0 regardless of clip_norm value
        nan_safe: If True, replace NaNs/Infs with zeros before clipping

    Returns:
        Tuple of (clipped_pytree, original_norm)

    Edge cases:
        - clip_norm=0, rescale=False: Returns zeros
        - clip_norm=0, rescale=True: Returns pytree/norm (unit norm)
        - clip_norm=inf, rescale=False: No clipping (passthrough)
        - clip_norm=inf, rescale=True: Returns zeros
        - pytree_norm=0: Returns unchanged
    """
    # Handle NaN/Inf
    if nan_safe:
        pytree = tree_map(lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), pytree)

    # Compute original norm
    orig_norm = global_norm(pytree)

    # Compute scale factor
    clip_norm_tensor = torch.tensor(clip_norm, dtype=orig_norm.dtype, device=orig_norm.device)
    clip_norm_tensor = torch.clamp(clip_norm_tensor, min=0.0)

    # Basic clipping: scale = min(1, clip_norm / orig_norm)
    scale = torch.minimum(torch.tensor(1.0), clip_norm_tensor / orig_norm)

    # Rescale to unit norm if requested
    if rescale_to_unit_norm:
        # If clip_norm > 0: scale / clip_norm
        # If clip_norm == 0: 1 / orig_norm
        scale = torch.where(clip_norm_tensor > 0, scale / clip_norm_tensor, 1.0 / orig_norm)

    # Handle norm=0 or NaN: set scale to 0
    scale = torch.where(torch.isfinite(scale), scale, torch.tensor(0.0))

    # Apply scale
    def scale_leaf(t):
        if not isinstance(t, torch.Tensor):
            return t
        return scale * t

    clipped = tree_map(lambda t: scale_leaf(t) if isinstance(t, torch.Tensor) else t, pytree)
    return clipped, orig_norm


def _with_extra_batch_axis(fun, batch_argnums):
    """Wraps a function to add an extra batch axis to the batch_argnums."""
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    def wrapped_fun(*args, **kwargs):
        args_with_group_axis = list(args)
        for i in batch_argnums:
            args_with_group_axis[i] = tree_map(lambda x: x.unsqueeze(1) if isinstance(x, torch.Tensor) else x, args[i])
        return fun(*args_with_group_axis, **kwargs)

    return wrapped_fun


def clip_sum(
    fun,
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    keep_batch_dim: bool = True,
    l2_clip_norm: float = 1.0,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    return_norms: bool = False,
    nan_safe: bool = True,
    dtype: torch.dtype | None = None,
):
    """Transform a function to clip its per-example outputs and sum across batch.

    This is the core operation for DP-SGD. Typically used with grad(loss_fn).

    Args:
        fun: Function whose outputs will be clipped and summed.
            Should return PyTree (or (PyTree, aux) if has_aux=True).
        has_aux: If True, fun returns (value, aux). Only value is clipped.
        batch_argnums: Which arguments have batch dimension (default: 0)
        keep_batch_dim: If True, pass batch args with size-1 batch dim to fun
        l2_clip_norm: Maximum L2 norm per example (default: 1.0)
        rescale_to_unit_norm: If True, scale by 1/clip_norm for unit sensitivity
        normalize_by: Divide summed result by this value (e.g., batch_size)
        return_norms: If True, return per-example norms before clipping
        nan_safe: If True, replace NaNs/Infs with zeros (default: True)
        dtype: Optional dtype for accumulation (e.g., torch.float64)

    Returns:
        BoundedSensitivityCallable: A wrapped function with same signature as fun.
        The callable's return signature depends on flags:
        - has_aux=False, return_norms=False: value
        - has_aux=True, return_norms=False: value, aux
        - has_aux=False, return_norms=True: value, norms
        - has_aux=True, return_norms=True: value, aux, norms

    Example:
        >>> # Clipped per-example gradient sum
        >>> def loss_fn(param, data):
        ...     return 0.5 * ((data - param) ** 2).mean()
        >>>
        >>> clipped_grad_fn = clip_sum(
        ...     torch.func.grad(loss_fn),
        ...     batch_argnums=1,
        ...     l2_clip_norm=1.0,
        ...     normalize_by=3.0
        ... )
        >>>
        >>> param = torch.tensor(3.0, requires_grad=True)
        >>> data = torch.tensor([0.0, 7.0, -2.0])
        >>> clipped_grad = clipped_grad_fn(param, data)
    """
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    # Wrap function to handle has_aux - use empty tuple () not None!
    if not has_aux:
        fun_with_aux = lambda *args, **kwargs: (fun(*args, **kwargs), ())
    else:
        fun_with_aux = fun

    def clipped_fn(*args, **kwargs):
        # Determine in_dims for vmap
        in_dims = tuple(0 if i in batch_argnums else None for i in range(len(args)))

        # Define per-example function
        def per_example_fn(*args_single):
            value, aux = fun_with_aux(*args_single, **kwargs)
            clipped_value, norm = clip_pytree(
                value, clip_norm=l2_clip_norm, rescale_to_unit_norm=rescale_to_unit_norm, nan_safe=nan_safe
            )
            return clipped_value, aux, norm

        # Vmap over batch - specify out_dims for aux
        # aux might be empty tuple (), which should have out_dims=None
        out_dims = (0, None if not has_aux else 0, 0)  # (clipped_value, aux, norm)
        vmapped = _vmap(per_example_fn, in_dims=in_dims, out_dims=out_dims)
        clipped_values, aux, norms = vmapped(*args)

        # Sum clipped values across batch dimension using tree_map
        # This handles both scalars/tensors and arbitrarily nested PyTrees
        result = tree_map(lambda x: torch.sum(x, dim=0, dtype=dtype), clipped_values)

        # Normalize
        if normalize_by != 1.0:
            result = tree_map(lambda x: x / normalize_by, result)

        # Return based on flags (matching JAX-Privacy's table)
        if not has_aux and not return_norms:
            return result
        elif has_aux and not return_norms:
            return result, aux
        elif not has_aux and return_norms:
            return result, norms
        else:  # has_aux and return_norms
            return result, aux, norms

    # Apply keep_batch_dim wrapper if needed
    if keep_batch_dim:
        clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)

    # Calculate sensitivity bound
    norm_bound = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by

    return BoundedSensitivityCallable(clipped_fn, norm_bound)
