"""Experimental per-example gradient clipping API.

Higher-level wrapper around clip_sum for gradient computation.
This module provides a convenient API similar to torch.func.grad but with
per-example gradient clipping for differential privacy.

References:
    JAX-Privacy gradient_clipping.py:
    https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/src/experimental/gradient_clipping.py
"""

from collections import namedtuple
from typing import Callable

import torch
from torch.func import grad as _torch_grad

from opaque.clipping import BoundedSensitivityCallable, clip_sum

# Named tuple for auxiliary outputs (matching JAX-Privacy)
AuxiliaryOutput = namedtuple("AuxiliaryOutput", ["values", "grad_norms", "aux"])


def _validate_static_args(argnums, batch_argnums, normalize_by):
    """Validates the argnums and batch_argnums inputs are compatible."""
    if normalize_by <= 0.0:
        raise ValueError(f"normalize_by must be > 0, got {normalize_by}.")
    if isinstance(argnums, int):
        argnums = (argnums,)
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)
    if not batch_argnums:
        raise ValueError("Batch argnums must not be empty.")
    if min(argnums + batch_argnums) < 0:
        raise ValueError(f"argnums={argnums} and batch_argnums={batch_argnums} must be >= 0.")
    shared_argnums = set(argnums) & set(batch_argnums)
    if shared_argnums:
        raise ValueError(
            "Cannot compute clipped gradients for argnums that have a batch axis. "
            f"{argnums=} and {batch_argnums=} with overlap {list(shared_argnums)}."
        )


def _validate_args(argnums, batch_argnums, args):
    """Validates the arguments to the per-example gradient clipping function."""
    if isinstance(argnums, int):
        argnums = (argnums,)
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)
    max_argnum = max(argnums + batch_argnums)
    if len(args) <= max_argnum:
        raise ValueError(f"Unable to find argnum={max_argnum}, was given {len(args)} args.")

    # Validate batch axis sizes are consistent
    batch_args = [args[i] for i in batch_argnums]
    batch_axis_sizes = set()
    for arg in batch_args:
        if isinstance(arg, torch.Tensor):
            batch_axis_sizes.add(arg.shape[0])
        elif isinstance(arg, dict):
            # For PyTree args, check first leaf
            for v in arg.values():
                if isinstance(v, torch.Tensor):
                    batch_axis_sizes.add(v.shape[0])
                    break
    if len(batch_axis_sizes) > 1:
        raise ValueError(
            f"Batch axis must have the same size for all inputs in batch_argnums, " f"got {batch_axis_sizes}."
        )


def clipped_grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    l2_clip_norm: float,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    keep_batch_dim: bool = True,
    return_values: bool = False,
    return_grad_norms: bool = False,
    pre_clipping_transform: Callable | None = None,
    nan_safe: bool = True,
    dtype: torch.dtype | None = None,
) -> BoundedSensitivityCallable:
    """Create a function to compute the sum of clipped gradients of fun.

    This function acts as a transformation similar to `torch.func.grad`, but with added
    functionality for gradient clipping applied on a per-example (or per-group)
    basis before summation. It computes the gradient of `fun` with respect to
    `argnums`, calculates the L2 norm of the gradient for each example slice
    along the first axis of the `batch_argnums` args, clips each per-example
    gradient to have a norm of at most `l2_clip_norm`, and finally sums these
    clipped gradients.

    Non-grad outputs of the returned function (aux, values, grad_norms) may
    optionally be returned by setting the arguments `has_aux`, `return_values`,
    and/or `return_grad_norms` to True. These outputs are per-example, and hence
    have a batch axis. It is up to the caller to handle these as necessary.

    Example Usage:
        >>> import torch
        >>> from opaque.gradient_clipping import clipped_grad
        >>> f = lambda param, data: 0.5 * ((data - param) ** 2).mean()
        >>> g = clipped_grad(f, l2_clip_norm=float('inf'))
        >>> g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]))
        tensor(1.3333)

    Formal Guarantees:
        For the gradient output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be 1.0 if
          `rescale_to_unit_norm` is True. Otherwise, the sensitivity is
          `l2_clip_norm`. Under replace-one DP, the sensitivity is doubled
          (2.0 or 2 * `l2_clip_norm`).
        All auxiliary outputs (aux, values, grad_norms) are per-example. This
          function guarantees that per-example outputs only depend on the data for the
          same example. This allows maximum flexibility for the caller to aggregate
          these as desired (possibly with a DP mean, median, quantile, or histogram
          mechanism).

    Args:
        fun: The function to be differentiated, which should return a scalar loss
            value. If `has_aux` is True, it should return a tuple `(value, aux)`.
        argnums: Specifies which argument(s) of `fun` to differentiate with respect
            to. Can be an integer or a sequence of integers. These arguments should
            *not* have a batch dimension.
        has_aux: If True, `fun` is expected to return a tuple `(value, aux)`. The
            auxiliary data `aux` will be returned by the transformed function.
            Exercise caution when using this as no DP sensitivity guarantees are
            provided for the auxiliary data.
        l2_clip_norm: The maximum L2 norm for each per-example gradient. Gradients
            with a norm larger than this value will be scaled down.
        rescale_to_unit_norm: If True, clipped gradients are rescaled by
            `1.0 / l2_clip_norm`. This ensures the sensitivity is 1.0. If False, they
            are only scaled down if their norm exceeds `l2_clip_norm`, resulting in a
            sensitivity of `l2_clip_norm`.
        normalize_by: Divide the clipped output by this value before returning.
        batch_argnums: Specifies which argument(s) of `fun` contain the batch
            dimension (usually the data and labels). Can be an integer or a sequence
            of integers. All arguments specified here must have the same size along
            their first dimension (the batch dimension). The default value of 1 assumes
            the signature of fun is `fun(params, batch)`.
        keep_batch_dim: If True, batch inputs will be passed to `fun` with a leading
            batch axis of size 1. If False, this size 1 axis will be dropped
            (reducing the rank of the batch args by 1 before passing to `fun`). The
            default value of True assumes that `fun` expects inputs with a batch axis.
        return_values: If True, the transformed function will also return the
            per-example values, before clipping.
        return_grad_norms: If True, the transformed function will also return the
            per-example gradient norms, before clipping.
        pre_clipping_transform: An optional function to apply to the per-example
            gradients before clipping. The function should consume the gradient pytree
            for a single example and return a new pytree (possibly with different
            structure). Can be used to e.g., scale the leaves of the pytree to
            accommodate preconditioner clipping. Does not affect the sensitivity
            guarantee. Default is identity function.
        nan_safe: If True, the formal guarantees of the returned Callable still
            hold in the presence of NaNs and infs.
        dtype: Optional dtype for the returned gradient. If None, the dtype will be
            the same as the dtypes of the gradient function. Can be useful to avoid
            overflow issues when using low-precision dtypes as the returned function
            computes a sum over a potentially large batch.

    Returns:
        A new function `clipped_grad_fn` that computes the sum of clipped
        per-example gradients of `fun`. The returned function returns `grad`
        if return_values = return_grad_norms = has_aux = False. Otherwise, it
        returns a tuple of (grad, AuxiliaryOutput), where AuxiliaryOutput is a
        namedtuple with optional fields (values, grad_norms, aux) containing the
        per-example values, gradient norms, and auxiliary data, respectively.
    """
    _validate_static_args(argnums, batch_argnums, normalize_by)

    # Default pre_clipping_transform to identity
    if pre_clipping_transform is None:
        pre_clipping_transform = lambda x: x

    # Create value_and_grad function (PyTorch doesn't have this built-in)
    # We need to compute both the loss value and gradient
    def value_and_grad_fn(*args, **kwargs):
        """Compute both value and gradient."""
        # For has_aux=False: fun returns scalar
        # For has_aux=True: fun returns (scalar, aux)
        if has_aux:

            def loss_fn_for_grad(*grad_args):
                val, _ = fun(*grad_args, **kwargs)
                return val

            # Compute gradient
            grad_result = _torch_grad(loss_fn_for_grad, argnums=argnums)(*args)

            # Also compute value and aux
            value, aux = fun(*args, **kwargs)
            return (value, aux), grad_result
        else:
            # Compute gradient
            grad_result = _torch_grad(fun, argnums=argnums)(*args, **kwargs)

            # Also compute value
            value = fun(*args, **kwargs)
            return value, grad_result

    # Create the grad function that returns (transformed_grad, value_and_aux)
    def grad_fn(*args, **kwargs):
        value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
        return pre_clipping_transform(grad), value_and_aux

    # Use clip_sum with has_aux=True and return_norms=True
    clipped_grad_fn = clip_sum(
        grad_fn,
        has_aux=True,
        batch_argnums=batch_argnums,
        l2_clip_norm=l2_clip_norm,
        keep_batch_dim=keep_batch_dim,
        rescale_to_unit_norm=rescale_to_unit_norm,
        normalize_by=normalize_by,
        return_norms=True,
        nan_safe=nan_safe,
        dtype=dtype,
    )

    # Wrap to unpack and format outputs
    def wrapped_clipped_grad_fn(*args, **kwargs):
        _validate_args(argnums, batch_argnums, args)
        grad, values_and_maybe_aux, norms = clipped_grad_fn(*args, **kwargs)

        # Unpack values and aux
        if has_aux:
            values = values_and_maybe_aux[0]
            aux = values_and_maybe_aux[1]
        else:
            values = values_and_maybe_aux
            aux = None

        # Build auxiliary output
        per_example_aux = AuxiliaryOutput(
            values=values if return_values else None,
            grad_norms=norms if return_grad_norms else None,
            aux=aux if has_aux else None,
        )

        # Return based on what user requested
        if has_aux or return_values or return_grad_norms:
            return grad, per_example_aux
        return grad

    return BoundedSensitivityCallable(wrapped_clipped_grad_fn, clipped_grad_fn.l2_norm_bound)
