"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping per-example outputs (typically gradients)
to bounded L2 norms, which is a core primitive for DP-SGD training.

This module consolidates functionality from JAX-Privacy's experimental/clipping.py
and experimental/gradient_clipping.py (which was deleted in main branch).

References:
    JAX-Privacy implementation:
    https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/experimental
"""

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import grad as _torch_grad
from torch.func import vmap as _vmap

from opaque.pytree_utils import global_norm, tree_map

# Named tuple for auxiliary outputs (matching JAX-Privacy)
AuxiliaryOutput = namedtuple("AuxiliaryOutput", ["values", "grad_norms", "aux"])


def _value_and_grad(fun: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False):
    """Create a function that returns both value and gradient, mimicking jax.value_and_grad.

    PyTorch's torch.func.grad returns (grad, aux) when has_aux=True,
    but JAX's jax.value_and_grad returns ((value, aux), grad).
    This helper provides JAX-compatible behavior.

    Args:
        fun: Function to differentiate. If has_aux=True, should return (value, aux).
        argnums: Arguments to differentiate with respect to.
        has_aux: Whether fun returns auxiliary data.

    Returns:
        A function that returns ((value, aux), grad) if has_aux=True,
        or (value, grad) if has_aux=False.
    """
    grad_fn = _torch_grad(fun, argnums=argnums, has_aux=has_aux)

    if has_aux:

        def wrapper(*args, **kwargs):
            # Call original function to get value and aux
            value, aux = fun(*args, **kwargs)
            # Get gradient (aux from grad is the same)
            gradient, _ = grad_fn(*args, **kwargs)
            # Return in JAX format: ((value, aux), grad)
            return (value, aux), gradient

    else:

        def wrapper(*args, **kwargs):
            value = fun(*args, **kwargs)
            gradient = grad_fn(*args, **kwargs)
            return value, gradient

    return wrapper


@dataclass(frozen=True)
class BoundedSensitivityCallable:
    """Callable with a sensitivity property.

    If has_aux is False, the sensitivity guarantee holds for the entire output
    which may be an arbitrary pytree of Tensors. If has_aux is True, the
    output of the function is a pair `(value, aux)` and the sensitivity guarantee
    only holds for `value` PyTree. The aux PyTree is returned on a per-example
    basis (i.e., as a PyTree of tensors having a batch axis). The caller should
    handle the aux output with care w.r.t. DP guarantees, should they be needed.
    """

    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def sensitivity(self, neighboring_relation: str = "REPLACE_SPECIAL") -> float:
        """Returns the L2 sensitivity of the Callable.

        The L2 sensitivity is defined with respect to the given neighboring relation
        and the unit of privacy implied by the function that created this instance.

        Args:
            neighboring_relation: The neighboring relation to consider. One of:
                - "ADD_OR_REMOVE_ONE": Dataset differs by adding/removing one record
                - "REPLACE_ONE": Dataset differs by replacing one record
                - "REPLACE_SPECIAL": Dataset differs by replacing one record with a special element

        Returns:
            The L2 sensitivity of the Callable.

        Raises:
            ValueError: If neighboring_relation is not supported.
        """
        if neighboring_relation == "ADD_OR_REMOVE_ONE":
            return self.l2_norm_bound
        elif neighboring_relation == "REPLACE_ONE":
            return 2 * self.l2_norm_bound
        elif neighboring_relation == "REPLACE_SPECIAL":
            return self.l2_norm_bound
        else:
            raise ValueError(f"Unsupported neighboring_relation={neighboring_relation}")


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
    nan_safe: bool = False,
    return_zero: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip a PyTree of tensors to a maximum L2 norm.

    Args:
        pytree: Dictionary of tensors to clip
        clip_norm: Maximum L2 norm (non-negative, or inf for no clipping)
        rescale_to_unit_norm: If True, additionally scale by 1/clip_norm
            so final norm is at most 1.0 regardless of clip_norm value
        nan_safe: If True, replace NaNs/Infs with zeros before clipping
        return_zero: If True, the output PyTree is guaranteed to be zero no matter
            what the inputs are. Does not influence the formal guarantees but useful
            for privacy amplification via padding (see https://arxiv.org/pdf/2411.04205).

    Returns:
        Tuple of (clipped_pytree, original_norm)

    Edge cases:
        - clip_norm=0, rescale=False: Returns zeros
        - clip_norm=0, rescale=True: Returns pytree/norm (unit norm)
        - clip_norm=inf, rescale=False: No clipping (passthrough)
        - clip_norm=inf, rescale=True: Returns zeros
        - pytree_norm=0: Returns unchanged
        - return_zero=True: Returns zeros regardless of other parameters
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

    # Apply return_zero if requested (for privacy amplification via padding)
    if return_zero:
        clipped = tree_map(
            lambda t: torch.zeros_like(t) if isinstance(t, torch.Tensor) else t, clipped
        )

    return clipped, orig_norm


def _with_extra_batch_axis(fun, batch_argnums):
    """Wraps a function to add an extra batch axis to the batch_argnums."""
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    def wrapped_fun(*args, **kwargs):
        args_with_group_axis = list(args)
        for i in batch_argnums:
            args_with_group_axis[i] = tree_map(
                lambda x: x.unsqueeze(1) if isinstance(x, torch.Tensor) else x, args[i]
            )
        return fun(*args_with_group_axis, **kwargs)

    return wrapped_fun


def clipped_fun(
    fun,
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    keep_batch_dim: bool = True,
    l2_clip_norm: float = 1.0,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    return_norms: bool = False,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: torch.dtype | None = None,
    spmd_axis_name: str | None = None,
):
    """Transform a function to clip its output and sum across a batch.

    This is the primary API for per-example clipping in DP-SGD. It wraps a function
    to clip each per-example output to a maximum L2 norm, then sums the clipped outputs.

    Example Usage:
        >>> data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        >>> clipped_mean = clipped_fun(torch.mean, l2_clip_norm=1.0)
        >>> clipped_mean(data)
        tensor(5.)

    Formal Guarantees:
        For the first function output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be 1.0 if
          `rescale_to_unit_norm` is True. Otherwise, the sensitivity is
          `l2_clip_norm`. Under replace-one DP, the sensitivity is doubled
          (2.0 or 2 * `l2_clip_norm`).
        Extra auxiliary outputs (aux, norms) are per-example. This function
          guarantees that per-example outputs only depend the data for the same
          example. This allows maximum flexibility for the caller to aggregate
          these as desired (possibly with a DP mean, median, quantile, or histogram
          mechanism).

    Args:
        fun: The function to be clipped.
        has_aux: If True, `fun` is expected to return a tuple `(value, aux)`. Only
            the value will be clipped + aggregated, `aux` will be returned on a
            per-example basis. Exercise caution when using this as the sensitivity
            guarantees of the returned Callable are only provided w.r.t. `value`.
        batch_argnums: Specifies which argument(s) of `fun` contain the batch
            dimension. All arguments specified here must have the same size along the
            0th axis.
        keep_batch_dim: If True, batch inputs will be passed to `fun` with a leading
            batch axis of size 1. If False, this size 1 axis will be dropped
            (reducing the rank of the batch args by 1 before passing to `fun`).
        l2_clip_norm: The maximum L2 norm allowed.
        rescale_to_unit_norm: If True, the output PyTree's norm is rescaled by `1.0
            / clip_norm` after potential clipping. If False, the output PyTree has
            norm at most `clip_norm`.
        normalize_by: Divide the clipped output by this value before returning.
        return_norms: If True, the returned Callable will return the l2_norms of the
            per-example values before clipping. These values should be handled with
            care, see the formal guarantees above.
        microbatch_size: If set, the batch is split up into microbatches of this
            size. **Currently not implemented** - parameter accepted for API
            compatibility but ignored. Will be implemented in future release.
        nan_safe: If True, the formal guarantees of the returned Callable still
            hold in the presence of NaNs and infs. See `clip_pytree` for more details.
        dtype: Optional dtype for the clipped+aggregated pytree. If None, the dtype
            will be the same as the dtypes of the function output.
        spmd_axis_name: See torch.vmap. **Currently not implemented** - parameter
            accepted for API compatibility.

    Returns:
        A new function `clip_fn` that clips the output of `fun` and sums across
        the batch. `clip_fn` takes the same arguments as `fun`. The exact output
        signature depends on `has_aux` and `return_norms`:

        | `has_aux` | `return_norms` | `clipped_fn` returns  |
        | :-------- | :--------------| :-------------------- |
        | `False`   | `False`        | `value`               |
        | `True`    | `False`        | `value, aux`          |
        | `False`   | `True`         | `value, norms`        |
        | `True`    | `True`         | `value, (aux, norms)` |

    Note:
        The output signature for `has_aux=True, return_norms=True` differs from
        the older `clip_sum()` function (which returns `value, aux, norms`).
        This matches JAX-Privacy main branch API.
    """
    # Warn about unimplemented parameters
    if microbatch_size is not None:
        import warnings

        warnings.warn(
            "microbatch_size parameter is not yet implemented and will be ignored. "
            "This is documented tech debt and will be added in a future release.",
            UserWarning,
        )
    if spmd_axis_name is not None:
        import warnings

        warnings.warn(
            "spmd_axis_name parameter is not yet implemented and will be ignored.",
            UserWarning,
        )

    # Normalize batch_argnums to tuple
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    # Wrap function to handle has_aux - use empty tuple () not None!
    if not has_aux:

        def fun_with_aux(*args, **kwargs):
            return (fun(*args, **kwargs), ())

    else:
        fun_with_aux = fun

    def clipped_fn(*args, **kwargs):
        # Determine in_dims for vmap
        in_dims = tuple(0 if i in batch_argnums else None for i in range(len(args)))

        # Define per-example function
        def per_example_fn(*args_single):
            value, aux = fun_with_aux(*args_single, **kwargs)
            clipped_value, norm = clip_pytree(
                value,
                clip_norm=l2_clip_norm,
                rescale_to_unit_norm=rescale_to_unit_norm,
                nan_safe=nan_safe,
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

        # Return based on flags (matching JAX-Privacy main branch output signature)
        if not has_aux and not return_norms:
            return result
        elif has_aux and not return_norms:
            return result, aux
        elif not has_aux and return_norms:
            # JAX-Privacy main: return (value, (aux, norms)) where aux=()
            return result, ((), norms)
        else:  # has_aux and return_norms
            # JAX-Privacy main: return (value, (aux, norms))
            return result, (aux, norms)

    # Apply keep_batch_dim wrapper if needed
    if keep_batch_dim:
        clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)

    # Calculate sensitivity bound
    norm_bound = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by

    # Determine if output has auxiliary data
    output_has_aux = has_aux or return_norms

    return BoundedSensitivityCallable(clipped_fn, norm_bound, output_has_aux)


# ============================================================================
# Helper functions for clipped_grad (from gradient_clipping.py)
# ============================================================================


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
            f"Batch axis must have the same size for all inputs in batch_argnums, "
            f"got {batch_axis_sizes}."
        )


# ============================================================================
# High-level gradient clipping API (from gradient_clipping.py)
# ============================================================================


def _normalize_fun_to_return_aux(fun: Callable, has_aux: bool) -> Callable:
    """Normalize function to always return (value, aux) tuple.

    Args:
        fun: Function to normalize.
        has_aux: Whether fun already returns aux.

    Returns:
        Normalized function that always returns (value, aux).
    """
    if has_aux:
        return fun
    else:
        return lambda *args, **kwargs: (fun(*args, **kwargs), ())


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
        microbatch_size: If set, input groups are formed into microbatches of this
            size. Not yet implemented in PyTorch version (tech debt).
        nan_safe: If True, the formal guarantees of the returned Callable still
            hold in the presence of NaNs and infs.
        dtype: Optional dtype for the returned gradient. If None, the dtype will be
            the same as the dtypes of the gradient function. Can be useful to avoid
            overflow issues when using low-precision dtypes as the returned function
            computes a sum over a potentially large batch.
        spmd_axis_name: Axis name for SPMD distributed training. Not yet implemented
            in PyTorch version (tech debt).

    Returns:
        A new function `clipped_grad_fn` that computes the sum of clipped
        per-example gradients of `fun`. The returned function returns `grad`
        if return_values = return_grad_norms = has_aux = False. Otherwise, it
        returns a tuple of (grad, AuxiliaryOutput), where AuxiliaryOutput is a
        namedtuple with optional fields (values, grad_norms, aux) containing the
        per-example values, gradient norms, and auxiliary data, respectively.
    """
    _validate_static_args(argnums, batch_argnums, normalize_by)
    fun = _normalize_fun_to_return_aux(fun, has_aux)

    # Use our _value_and_grad helper to get JAX-compatible behavior
    value_and_grad_fn = _value_and_grad(fun, argnums=argnums, has_aux=True)

    def grad_fn(*args, **kwargs):
        value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
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

    clipped_grad_fn = clipped_fun(
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

    # Wrap the result to convert dict to AuxiliaryOutput
    if not (has_aux or return_values or return_grad_norms):
        # No aux, return directly
        return clipped_grad_fn
    else:
        # Need to convert aux_dict to AuxiliaryOutput
        def wrapper(*args, **kwargs):
            grad, aux_dict = clipped_grad_fn(*args, **kwargs)
            aux_output = AuxiliaryOutput(
                values=aux_dict.get("values"),
                grad_norms=aux_dict.get("grad_norms"),
                aux=aux_dict.get("aux"),
            )
            return grad, aux_output

        # Return wrapped function with same properties
        return BoundedSensitivityCallable(
            wrapper,
            clipped_grad_fn.l2_norm_bound,
            clipped_grad_fn.has_aux,
        )
