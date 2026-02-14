"""Per-example gradient clipping for differential privacy."""

from collections.abc import Callable

import torch
from torch.func import grad_and_value

from opaque.clipping._helpers import normalize_fun_to_return_aux, normalize_to_tuple
from opaque.clipping.clipped_fun import clipped_fun
from opaque.clipping.types import ClippedGradAux
from opaque.utils.pytree import global_norm


def _validate_static_args(argnums, batch_argnums, normalize_by):
    """Validates the argnums and batch_argnums inputs are compatible."""
    if normalize_by <= 0.0:
        raise ValueError(f"normalize_by must be > 0, got {normalize_by}.")
    argnums = normalize_to_tuple(argnums)
    batch_argnums = normalize_to_tuple(batch_argnums)
    if not batch_argnums:
        raise ValueError("Batch argnums must not be empty.")
    if min(argnums + batch_argnums) < 0:
        raise ValueError(
            f"argnums={argnums} and batch_argnums={batch_argnums} must be >= 0."
        )
    shared_argnums = set(argnums) & set(batch_argnums)
    if shared_argnums:
        raise ValueError(
            "Cannot compute clipped gradients for argnums that have a batch axis. "
            f"{argnums=} and {batch_argnums=} with overlap {list(shared_argnums)}."
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
    pre_clipping_transform: Callable = lambda x: x,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: torch.dtype | None = None,
    spmd_axis_name: str | None = None,
) -> Callable:
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
        >>> from opaque.clipping import clipped_grad
        >>> f = lambda param, data: 0.5 * ((data - param) ** 2).mean()
        >>> g = clipped_grad(f, l2_clip_norm=float('inf'))
        >>> g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]))
        tensor(1.3333)

    Example Usage (with Auxiliary Output):
        >>> g = clipped_grad(
        ...     f, l2_clip_norm=float('inf'), return_values=True, return_grad_norms=True
        ... )
        >>> _, aux = g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]))
        >>> aux.values
        tensor([4.5000, 8.0000, 12.5000])
        >>> aux.grad_norms
        tensor([3., 4., 5.])

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
        microbatch_size: If set, the batch is split up into microbatches of this
            size for memory-efficient processing. Processes each microbatch separately
            and accumulates results without materializing the full batch of gradients.
            Set this to reduce peak memory usage at the cost of slightly slower computation.
        nan_safe: If True, the formal guarantees of the returned Callable still
            hold in the presence of NaNs and infs.
        dtype: Optional dtype for the returned gradient. If None, the dtype will be
            the same as the dtypes of the gradient function. Can be useful to avoid
            overflow issues when using low-precision dtypes as the returned function
            computes a sum over a potentially large batch.
        spmd_axis_name: Axis name for SPMD distributed training. Not yet implemented
            in PyTorch version (tech debt).

    Returns:
        Tuple of (clipped_grad_fn, clip_state) where:
        - clipped_grad_fn: A function that computes the sum of clipped per-example gradients.
          Call signature: clipped_grads, new_state = clipped_grad_fn(..., state=clip_state)
          If auxiliary outputs are requested, returns: (clipped_grads, grad_aux), new_state
        - clip_state: Initial FixedClipState containing sensitivity information

        The grad_aux output (when requested) is a ClippedGradAux named tuple with fields:
            - loss_values: Per-example function values (if return_values=True), else None
            - grad_norms: Per-example gradient norms (if return_grad_norms=True), else None
            - user_aux: Per-example auxiliary data (if has_aux=True), else None
    """
    _validate_static_args(argnums, batch_argnums, normalize_by)
    fun = normalize_fun_to_return_aux(fun, has_aux)

    # Use PyTorch's grad_and_value (returns (grad, value) or (grad, (value, aux)))
    grad_and_value_fn = grad_and_value(fun, argnums=argnums, has_aux=True)

    def grad_fn(*args, **kwargs):
        grad, value_and_aux = grad_and_value_fn(*args, **kwargs)
        result = pre_clipping_transform(grad)
        if has_aux or return_values or return_grad_norms:
            # Return a dict instead of AuxiliaryOutput to avoid vmap issues with None values
            # PyTorch vmap cannot handle namedtuples with None values when out_dims != None
            aux_dict = {}
            if return_values:
                aux_dict["values"] = value_and_aux[0]
            if return_grad_norms:
                aux_dict["grad_norms"] = global_norm(grad)
            if has_aux:
                aux_dict["aux"] = value_and_aux[1]
            return result, aux_dict
        return result

    clipped_grad_fn, clip_state = clipped_fun(
        grad_fn,
        has_aux=has_aux or return_values or return_grad_norms,
        batch_argnums=batch_argnums,
        l2_clip_norm=l2_clip_norm,
        keep_batch_dim=keep_batch_dim,
        rescale_to_unit_norm=rescale_to_unit_norm,
        normalize_by=normalize_by,
        microbatch_size=microbatch_size,
        nan_safe=nan_safe,
        dtype=dtype,
        spmd_axis_name=spmd_axis_name,
    )

    # clipped_grad_fn is now a callable, clip_state is a FixedClipState
    # Wrap the result to convert dict to AuxiliaryOutput
    if not (has_aux or return_values or return_grad_norms):
        # No aux, return wrapped directly with state-passing signature
        def grad_fn_wrapper(*args, state, **kwargs):
            (result, returned_state) = clipped_grad_fn(*args, state=state, **kwargs)
            return result, returned_state

        return grad_fn_wrapper, clip_state
    else:
        # Need to convert aux_dict to ClippedGradAux
        def grad_fn_wrapper(*args, state, **kwargs):
            (clipped_grads, aux_dict), returned_state = clipped_grad_fn(
                *args, state=state, **kwargs
            )
            grad_aux = ClippedGradAux(
                loss_values=aux_dict.get("values"),
                grad_norms=aux_dict.get("grad_norms"),
                user_aux=aux_dict.get("aux"),
            )
            return (clipped_grads, grad_aux), returned_state

        return grad_fn_wrapper, clip_state


__all__ = ["clipped_grad"]
