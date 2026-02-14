"""Per-example clipping and summing for arbitrary functions."""

from collections.abc import Callable

import torch
from torch.func import vmap as _vmap

from opaque.clipping._helpers import normalize_to_tuple
from opaque.clipping.pytree import clip_pytree
from opaque.clipping.types import FixedClipState
from opaque.utils.pytree import tree_map


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
) -> Callable:
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
            size for memory-efficient processing. Unlike PyTorch's vmap chunk_size,
            this implementation processes chunks sequentially and accumulates the
            sum incrementally, ensuring only microbatch_size gradients are in memory
            at any time. This is crucial for DP-SGD where per-example gradients are
            the memory bottleneck. Set this to reduce peak memory usage at the cost
            of slightly slower computation.
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
    if spmd_axis_name is not None:
        import warnings

        warnings.warn(
            "spmd_axis_name parameter is not yet implemented and will be ignored.",
            UserWarning,
        )

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
                rescale_to_unit_norm=rescale_to_unit_norm,
                nan_safe=nan_safe,
            )
            return clipped_value, aux, norm

        # Get batch size from first batch argument
        first_batch_arg_idx = batch_argnums[0]
        first_batch_arg = args[first_batch_arg_idx]
        # Handle PyTree batch args by getting first leaf
        if isinstance(first_batch_arg, (dict, list, tuple)):
            from opaque.utils.pytree import tree_leaves
            batch_size = tree_leaves(first_batch_arg)[0].shape[0]
        else:
            batch_size = first_batch_arg.shape[0]

        # Choose between microbatched and full-batch processing
        if microbatch_size is None or microbatch_size >= batch_size:
            # Full-batch processing (original behavior)
            out_dims = (0, None if not has_aux else 0, 0)  # (clipped_value, aux, norm)
            vmapped = _vmap(
                per_example_fn,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness="same",
            )
            clipped_values, aux, norms = vmapped(*args)

            # Sum clipped values across batch dimension
            result = tree_map(lambda x: torch.sum(x, dim=0, dtype=dtype), clipped_values)
        else:
            # True microbatching: process chunks and accumulate sum incrementally
            # This saves memory by not materializing all per-example outputs at once
            result = None
            all_aux = [] if has_aux else None
            all_norms = []

            # Process batch in chunks
            for chunk_start in range(0, batch_size, microbatch_size):
                chunk_end = min(chunk_start + microbatch_size, batch_size)

                # Extract chunk from batch arguments
                chunk_args = list(args)
                for i in batch_argnums:
                    chunk_args[i] = tree_map(
                        lambda x: x[chunk_start:chunk_end] if isinstance(x, torch.Tensor) else x,
                        args[i]
                    )

                # Process chunk with vmap (no chunk_size needed here - processing small chunk)
                out_dims = (0, None if not has_aux else 0, 0)
                vmapped = _vmap(
                    per_example_fn,
                    in_dims=in_dims,
                    out_dims=out_dims,
                    randomness="same",
                )
                chunk_clipped_values, chunk_aux, chunk_norms = vmapped(*chunk_args)

                # Sum chunk's clipped values and accumulate
                chunk_sum = tree_map(lambda x: torch.sum(x, dim=0, dtype=dtype), chunk_clipped_values)
                if result is None:
                    result = chunk_sum
                else:
                    # Add chunk sum to accumulated result
                    result = tree_map(lambda x, y: x + y, result, chunk_sum)

                # Collect auxiliary outputs
                if has_aux:
                    all_aux.append(chunk_aux)
                all_norms.append(chunk_norms)

            # Concatenate auxiliary outputs across chunks
            if has_aux:
                # Concatenate along batch dimension
                aux = tree_map(lambda *xs: torch.cat(xs, dim=0), *all_aux)
            else:
                aux = ()
            norms = torch.cat(all_norms, dim=0)

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

    # Calculate L2 sensitivity bound
    l2_norm_bound = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by

    # Create fixed clip state
    clip_state = FixedClipState(
        l2_norm_bound=l2_norm_bound,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )

    # Wrap function to accept and return state
    def stateful_clipped_fn(*args, state, **kwargs):
        result = clipped_fn(*args, **kwargs)
        return result, state  # State unchanged for fixed clipping

    # Return wrapped function with state
    return stateful_clipped_fn, clip_state


__all__ = ["clipped_fun"]
