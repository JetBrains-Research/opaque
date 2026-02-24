"""Per-example clipping and summing for arbitrary functions."""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch.func import vmap as _vmap

from opaque.clipping._helpers import normalize_to_tuple
from opaque.clipping.pytree import clip_pytree
from opaque.clipping.types import FixedClipState
from opaque.utils.pytree import global_norm, tree_map


class ClippedFunAux(NamedTuple):
    """Function-level auxiliary outputs from clipped_fun.

    Fields:
        loss_values: Per-example function values before clipping.
        grad_norms: Per-example L2 norms before clipping.
        clipped_grad_norms: Per-example L2 norms after clipping.
        loss_aux: Per-example auxiliary payload returned by the wrapped function.
    """

    loss_values: Any | None
    grad_norms: Any | None
    clipped_grad_norms: Any | None
    loss_aux: Any | None


def _with_extra_batch_axis(fun, batch_argnums):
    """Wraps a function to add an extra batch axis to the batch_argnums."""
    batch_argnums = normalize_to_tuple(batch_argnums)

    def wrapped_fun(*args, **kwargs):
        args_with_group_axis = list(args)
        for i in batch_argnums:
            args_with_group_axis[i] = tree_map(
                lambda x: x.unsqueeze(1) if isinstance(x, torch.Tensor) else x, args[i]
            )
        return fun(*args_with_group_axis, **kwargs)

    return wrapped_fun


def _microbatch_accumulate(
    per_example_fn,
    args,
    batch_argnums,
    in_dims,
    microbatch_size,
    return_aux,
    dtype,
):
    """Process batch in microbatches, accumulating results without materializing full batch.

    This implementation processes the batch in chunks of `microbatch_size`, accumulating
    results according to their type:
    - Clipped values: SUM (accumulate in-place, don't keep all per-example values)
    - Auxiliary outputs: CONCAT (keep per-example for privacy analysis)

    Args:
        per_example_fn: Function to vmap over each example
        args: Full batch arguments
        batch_argnums: Which arguments contain batch dimension
        in_dims: Input dimensions for vmap
        microbatch_size: Size of each microbatch
        return_aux: Whether function returns auxiliary outputs
        dtype: Optional dtype for accumulated gradients

    Returns:
        Tuple of (accumulated_values, concatenated_aux)
    """
    # Get batch size from first batch argument
    first_batch_idx = batch_argnums[0]
    first_batch_arg = args[first_batch_idx]
    if isinstance(first_batch_arg, torch.Tensor):
        batch_size = first_batch_arg.shape[0]
    else:
        # Handle PyTree case - get batch size from first tensor
        def get_first_tensor(pytree):
            if isinstance(pytree, torch.Tensor):
                return pytree
            elif isinstance(pytree, dict):
                for v in pytree.values():
                    result = get_first_tensor(v)
                    if result is not None:
                        return result
            elif isinstance(pytree, (list, tuple)):
                for v in pytree:
                    result = get_first_tensor(v)
                    if result is not None:
                        return result
            return None

        first_tensor = get_first_tensor(first_batch_arg)
        if first_tensor is None:
            raise ValueError(
                "Could not determine batch size: no torch.Tensor found in the "
                f"batch argument PyTree at index {first_batch_idx}."
            )
        batch_size = first_tensor.shape[0]

    # Initialize accumulators
    accumulated_grads = None
    aux_list = []

    # Process each microbatch
    for start_idx in range(0, batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, batch_size)

        # Slice batch arguments for this microbatch
        microbatch_args = list(args)
        for i in batch_argnums:
            microbatch_args[i] = tree_map(
                lambda x, s=start_idx, e=end_idx: (
                    x[s:e] if isinstance(x, torch.Tensor) else x
                ),
                args[i],
            )

        # vmap over microbatch
        if return_aux:
            out_dims = (0, 0)  # (clipped_value, aux)
            vmapped = _vmap(
                per_example_fn,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness="same",
            )
            clipped_values, aux = vmapped(*microbatch_args)
        else:
            out_dims = 0  # clipped_value
            vmapped = _vmap(
                per_example_fn,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness="same",
            )
            clipped_values = vmapped(*microbatch_args)
            aux = ()

        # Accumulate clipped gradients (SUM)
        microbatch_sum = tree_map(
            lambda x: torch.sum(x, dim=0, dtype=dtype), clipped_values
        )
        if accumulated_grads is None:
            accumulated_grads = microbatch_sum
        else:
            accumulated_grads = tree_map(
                lambda acc, new: acc + new, accumulated_grads, microbatch_sum
            )

        # Collect aux outputs (CONCAT) - keep per-example
        if return_aux:
            aux_list.append(aux)

    # Concatenate aux across all microbatches
    if return_aux:
        # Concatenate aux outputs along batch dimension
        # Need to handle the list structure properly - transpose list of pytrees into pytree of lists
        def concat_leaves(*leaf_values):
            """Concatenate corresponding leaf values across microbatches."""
            if all(isinstance(v, torch.Tensor) for v in leaf_values):
                return torch.cat(leaf_values, dim=0)
            # Non-tensor leaves are assumed to be identical across microbatches;
            # return a single representative value to preserve the original structure.
            return leaf_values[0]

        aux = tree_map(concat_leaves, *aux_list)
    else:
        aux = ()

    return accumulated_grads, aux


def clipped_fun(
    fun,
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    keep_batch_dim: bool = True,
    l2_clip_norm: float = 1.0,
    normalize_by: float = 1.0,
    return_aux: bool = False,
    microbatch_size: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Callable, FixedClipState]:
    """Transform a function to clip its output and sum across a batch.

    This is the primary API for per-example clipping in DP-SGD. It wraps a function
    to clip each per-example output to a maximum L2 norm, then sums the clipped outputs.

    Example Usage:
        >>> data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        >>> clipped_mean, clip_state = clipped_fun(torch.mean, l2_clip_norm=1.0)
        >>> result, clip_state = clipped_mean(data, state=clip_state)
        >>> result
        tensor(5.)

    Formal Guarantees:
        For the first function output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be `l2_clip_norm`.
          Under replace-one DP, the sensitivity is doubled (2 * `l2_clip_norm`).
        Extra auxiliary outputs (aux, norms) are per-example. This function
          guarantees that per-example outputs only depend on the data for the same
          example. This allows maximum flexibility for the caller to aggregate
          these as desired (possibly with a DP mean, median, quantile, or histogram
          mechanism).

    Args:
        fun: The function to be clipped.
        has_aux: If True, `fun` is expected to return a tuple `(value, loss_aux)`. Only
            the value will be clipped + aggregated, `loss_aux` will be returned on a
            per-example basis. Exercise caution when using this as the sensitivity
            guarantees of the returned Callable are only provided w.r.t. `value`.
        batch_argnums: Specifies which argument(s) of `fun` contain the batch
            dimension. All arguments specified here must have the same size along the
            0th axis.
        keep_batch_dim: If True, batch inputs will be passed to `fun` with a leading
            batch axis of size 1. If False, this size 1 axis will be dropped
            (reducing the rank of the batch args by 1 before passing to `fun`).
        l2_clip_norm: The maximum L2 norm allowed.
        normalize_by: Divide the clipped output by this value before returning.
        return_aux: If True, the returned Callable will return a per-example aux
            NamedTuple containing the original per-example values, per-example norms
            before clipping, and any auxiliary data returned by `fun`.
        microbatch_size: If set, the batch is split up into microbatches of this
            size for memory-efficient processing. Processes each microbatch separately
            and accumulates results without materializing the full batch of gradients.
            Set this to reduce peak memory usage at the cost of slightly slower computation.
        dtype: Optional dtype for the clipped+aggregated pytree. If None, the dtype
            will be the same as the dtypes of the function output.
    Returns:
        A new function `clip_fn` that clips the output of `fun` and sums across
        the batch. `clip_fn` takes the same arguments as `fun`. The exact output
        signature depends on `return_aux`:

        | `return_aux` | `clipped_fn` returns  |
        | :----------- | :-------------------- |
        | `False`      | `value`               |
        | `True`       | `value, aux`          |
    """
    # Normalize batch_argnums to tuple
    batch_argnums = normalize_to_tuple(batch_argnums)

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
            )
            if return_aux:
                # Build aux dict with clipping metadata
                # IMPORTANT: Detach all tensors to prevent memory leaks from retaining
                # computational graphs. These are monitoring values, not used for gradients.
                aux_dict = {
                    "grad_norms": norm.norm.detach(),
                    "clipped_grad_norms": global_norm(clipped_value).detach(),
                }

                # Extract nested values and aux from wrapped functions (e.g., grad_fn)
                # aux may be a dict like {"loss_values": loss, "loss_aux": user_aux} or just user_aux
                if isinstance(aux, dict):
                    # Preserve "loss_values" from nested dict if present (e.g., loss from grad_and_value)
                    if "loss_values" in aux:
                        loss_val = aux["loss_values"]
                        aux_dict["loss_values"] = loss_val.detach() if isinstance(loss_val, torch.Tensor) else loss_val
                    else:
                        # No nested "loss_values", use function output
                        aux_dict["loss_values"] = value.detach() if isinstance(value, torch.Tensor) else value

                    # Extract user aux from nested dict if present
                    if has_aux:
                        if "loss_aux" in aux:
                            aux_dict["loss_aux"] = aux["loss_aux"]
                        else:
                            # aux is already the user aux (not nested)
                            aux_dict["loss_aux"] = aux
                else:
                    # aux is not a dict (direct user aux or None)
                    aux_dict["loss_values"] = value.detach() if isinstance(value, torch.Tensor) else value
                    if has_aux:
                        aux_dict["loss_aux"] = aux

                return clipped_value, aux_dict
            return clipped_value

        # Choose execution path based on microbatch_size
        if microbatch_size is None:
            # Fast path: vmap entire batch at once
            out_dims = 0 if not return_aux else (0, 0)  # (clipped_value, aux)
            vmapped = _vmap(
                per_example_fn,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness="same",
            )
            if return_aux:
                clipped_values, aux = vmapped(*args)
            else:
                clipped_values = vmapped(*args)
                aux = ()

            # Sum clipped values across batch dimension
            result = tree_map(
                lambda x: torch.sum(x, dim=0, dtype=dtype), clipped_values
            )
        else:
            # Manual microbatch accumulation: process in chunks, accumulate as we go
            result, aux = _microbatch_accumulate(
                per_example_fn=per_example_fn,
                args=args,
                batch_argnums=batch_argnums,
                in_dims=in_dims,
                microbatch_size=microbatch_size,
                return_aux=return_aux,
                dtype=dtype,
            )

        # Normalize
        if normalize_by != 1.0:
            result = tree_map(lambda x: x / normalize_by, result)

        if not return_aux:
            return result

        aux_dict = aux if isinstance(aux, dict) else {}
        aux = ClippedFunAux(
            loss_values=aux_dict.get("loss_values"),
            grad_norms=aux_dict.get("grad_norms"),
            clipped_grad_norms=aux_dict.get("clipped_grad_norms"),
            loss_aux=aux_dict.get("loss_aux"),
        )

        return result, aux

    # Apply keep_batch_dim wrapper if needed
    if keep_batch_dim:
        clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)

    # Calculate L2 sensitivity bound
    l2_norm_bound = l2_clip_norm / normalize_by

    # Create fixed clip state
    clip_state = FixedClipState(
        l2_norm_bound=l2_norm_bound,
    )

    # Wrap function to accept and return state
    def stateful_clipped_fn(*args, state, **kwargs):
        result = clipped_fn(*args, **kwargs)
        return result, state  # State unchanged for fixed clipping

    # Return wrapped function with state
    return stateful_clipped_fn, clip_state


__all__ = ["clipped_fun", "ClippedFunAux"]
