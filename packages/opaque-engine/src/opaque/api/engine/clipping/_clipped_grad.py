"""Per-example gradient clipping for differential privacy."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.autograd.profiler import record_function
from torch.func import grad_and_value

from opaque.api.engine.clipping._clipped_fun import FixedClipState, clipped_fun
from opaque.api.engine.clipping._helpers import (
    batch_size_from_args,
    normalize_fun_to_return_aux,
    normalize_to_tuple,
    zero_grads_like,
)
from opaque.api.engine.functional._transform_stack import (
    under_differentiating_transform,
)
from opaque.api.engine.pytree import tree_flatten
from opaque.api.engine.types import PerGroup, clipped

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.clipping._types import ClippedGradFn


@dataclass(frozen=True)
class ClippedGradAux:
    """Diagnostic outputs from clipped_grad.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use the
    returned ``ClippedPytree.max_norm`` metadata for noise calibration.

    Fields:
        loss_values: Per-example loss values before clipping.
        grad_norms: Per-example gradient L2 norms before clipping.
        clipped_grad_norms: Per-example gradient L2 norms after clipping.
        loss_aux: Per-example auxiliary payload returned by the loss function.
        clipping_rate: Fraction of per-example gradients whose norm exceeded
            the clipping threshold.
        batch_size: Number of examples in the batch.
        group_norms: Per-group per-example gradient L2 norms before clipping
            (dict[str, Tensor] with shape [batch_size] per group), or None
            when global clipping is used.
    """

    loss_values: Any | None = None
    grad_norms: Any | None = None
    clipped_grad_norms: Any | None = None
    loss_aux: Any | None = None
    clipping_rate: float | None = None
    batch_size: int = 0
    group_norms: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ClippedGradStatus:
    """Status for a clipped-gradient attempt.

    ``grads_were_finite`` is computed after ``pre_clipping_transform`` and
    before clipping replaces non-finite values with zero. Use it to update a
    loss scale while continuing the noised update and accounting on every step.
    """

    grads_were_finite: bool


def _pytree_finite_indicator(value: Any) -> torch.Tensor:
    """Return a scalar bool tensor without escaping a ``vmap`` transform."""
    leaves, _ = tree_flatten(value)
    indicator: torch.Tensor | None = None
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            continue
        finite = (
            torch.isfinite(leaf).all()
            if leaf.is_floating_point() or leaf.is_complex()
            else torch.ones((), dtype=torch.bool, device=leaf.device)
        )
        indicator = finite if indicator is None else indicator & finite
    if indicator is None:
        raise TypeError(
            "Expected at least one tensor leaf when checking gradient finiteness."
        )
    return indicator


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
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    clipping_norm: float | PerGroup,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    return_aux: bool = False,
    second_moment: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    microbatch_size: int | None = None,
    dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
    return_pre_clipping_finite: bool = False,
    _force_grad_norms: bool = False,
    _scale_fn: Callable | None = None,
    _return_stats: bool = False,
) -> tuple[ClippedGradFn, FixedClipState]:
    """Create a function to compute the sum of clipped gradients of loss_fn.

    This function acts as a transformation similar to `torch.func.grad`, but with added
    functionality for gradient clipping applied on a per-example (or per-group)
    basis before summation. It computes the gradient of `loss_fn` with respect to
    `argnums`, calculates the L2 norm of the gradient for each example slice
    along the first axis of the `batch_argnums` args, clips each per-example
    gradient to have a norm of at most `clipping_norm`, and finally sums these
    clipped gradients.

    Non-grad outputs of the returned function (aux) may optionally be returned
    by setting `return_aux=True`. These outputs are per-example, and hence have
    a batch axis. It is up to the caller to handle these as necessary.

    Example Usage:
        >>> import torch
        >>> from opaque.dpsgd.clipping import clipped_grad
        >>> f = lambda param, data: 0.5 * ((data - param) ** 2).mean()
        >>> g, clip_state = clipped_grad(f, clipping_norm=float('inf'))
        >>> result, clip_state = g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]), state=clip_state)
        >>> result
        tensor(1.3333)

    Example Usage (with Auxiliary Output):
        >>> g, clip_state = clipped_grad(
        ...     f, clipping_norm=float('inf'), return_aux=True
        ... )
        >>> (_, aux), clip_state = g(torch.tensor(3.0), torch.tensor([0.0, 7.0, -2.0]), state=clip_state)
        >>> aux.loss_values
        tensor([4.5000, 8.0000, 12.5000])
        >>> aux.grad_norms
        tensor([3., 4., 5.])

    Formal Guarantees:
        For the gradient output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be `clipping_norm`.
          Under replace-one DP, the sensitivity is doubled (2 * `clipping_norm`).
        All auxiliary outputs (loss_values, grad_norms) are per-example. This
          function guarantees that per-example outputs only depend on the data for the
          same example. This allows maximum flexibility for the caller to aggregate
          these as desired (possibly with a DP mean, median, quantile, or histogram
          mechanism).

    Args:
        loss_fn: The loss function to be differentiated, which should return a scalar
            value. If `has_aux` is True, it should return a tuple `(value, loss_aux)`.
        argnums: Specifies which argument(s) of `loss_fn` to differentiate with respect
            to. Can be an integer or a sequence of integers. These arguments should
            *not* have a batch dimension.
        has_aux: If True, `loss_fn` is expected to return a tuple `(value, loss_aux)`.
            The auxiliary data `loss_aux` will be returned by the transformed function.
            Exercise caution when using this as no DP sensitivity guarantees are
            provided for the auxiliary data.
        clipping_norm: The maximum L2 norm for each per-example gradient. Gradients
            with a norm larger than this value will be scaled down.
        normalize_by: Divide the clipped sum by this constant before returning.
            Set to expected batch size to produce averaged gradients with
            sensitivity = clipping_norm / normalize_by.
        batch_argnums: Specifies which argument(s) of `loss_fn` contain the batch
            dimension (usually the data and labels). Can be an integer or a sequence
            of integers. All arguments specified here must have the same size along
            their first dimension (the batch dimension). The default value of 1 assumes
            the signature of loss_fn is `loss_fn(params, batch)`.
        return_aux: If True, the transformed function will also return a per-example
            aux dataclass containing loss values, gradient norms, and loss aux.
        pre_clipping_transform: An optional function to apply to the per-example
            gradients before clipping. The function should consume the gradient pytree
            for a single example and return a new pytree (possibly with different
            structure). Can be used to e.g., scale the leaves of the pytree to
            accommodate preconditioner clipping. Account separately for any
            data-dependent control flow or telemetry. Default is identity function.
        microbatch_size: If set, the batch is split up into microbatches of this
            size for memory-efficient processing. Processes each microbatch separately
            and accumulates results without materializing the full batch of gradients.
            Set this to reduce peak memory usage at the cost of slightly slower computation.
        dtype: Optional dtype for the returned gradient. If None, the dtype will be
            the same as the dtypes of the gradient function. Can be useful to avoid
            overflow issues when using low-precision dtypes as the returned function
            computes a sum over a potentially large batch.
        compute_dtype: Internal accumulation dtype for reductions (per-example
            clip-norm and the across-batch sum). ``None`` (default) auto-promotes
            bf16/fp16 to float32 for numerical stability. Independent of
            ``dtype`` (which controls the *output* dtype).
        return_pre_clipping_finite: When True, return a
            :class:`ClippedGradStatus` beside the clipped result. Its
            ``grads_were_finite`` flag is measured after
            ``pre_clipping_transform`` and before clipping sanitizes NaN/Inf
            values. Use it for loss-scale backoff while continuing the noised
            update and accountant on every step.
        second_moment: Whether to accumulate the clipped-gradient second
            moment required by DP-FTRL noise mechanisms.
    Returns:
        Tuple of (:class:`ClippedGradFn`, clip_state) where:
        - clipped_grad_fn: A function that computes the sum of clipped per-example gradients.
          Call signature: clipped_grads, new_state = clipped_grad_fn(..., state=clip_state)
          If auxiliary outputs are requested, returns: (clipped_grads, grad_aux), new_state
          If ``return_pre_clipping_finite=True``, a
          :class:`ClippedGradStatus` is appended to that result.
        - clip_state: Initial FixedClipState containing sensitivity information

        The grad_aux output (when requested) is a ClippedGradAux dataclass with fields:
            - loss_values: Per-example function values (if return_aux=True)
            - grad_norms: Per-example gradient norms (if return_aux=True)
            - loss_aux: Per-example auxiliary data (if has_aux=True)
    """
    _validate_static_args(argnums, batch_argnums, normalize_by)
    if return_aux and _return_stats:
        raise ValueError("_return_stats cannot be combined with return_aux=True")
    if return_pre_clipping_finite and (_force_grad_norms or _return_stats):
        raise ValueError(
            "return_pre_clipping_finite cannot be combined with internal statistics options."
        )

    argnums_tuple = normalize_to_tuple(argnums)
    batch_argnums_tuple = normalize_to_tuple(batch_argnums)
    loss_fn = normalize_fun_to_return_aux(loss_fn, has_aux)

    output_max_norm = clipping_norm / normalize_by
    output_squared_max_norm = (
        (clipping_norm * clipping_norm) / normalize_by if second_moment else None
    )

    def _with_status(result: Any, grads_were_finite: bool) -> Any:
        if not return_pre_clipping_finite:
            return result
        status = ClippedGradStatus(grads_were_finite=grads_were_finite)
        return (*result, status) if isinstance(result, tuple) else (result, status)

    def _empty_batch_response(args, state):
        """Short-circuit for empty batches: zero grads + empty aux, no vmap."""
        zeros = zero_grads_like(args, argnums_tuple)
        if second_moment:
            from opaque.api.engine.types import SecondMomentClippingOutput

            grads = SecondMomentClippingOutput(
                grads=clipped(zeros, max_norm=output_max_norm),
                squared_grads=clipped(
                    zero_grads_like(args, argnums_tuple),
                    max_norm=output_squared_max_norm,
                ),
            )
        else:
            grads = clipped(zeros, max_norm=output_max_norm)
        if return_aux or _force_grad_norms:
            empty = torch.empty(0)
            grad_aux = ClippedGradAux(
                loss_values=empty if return_aux else None,
                grad_norms=empty,
                clipped_grad_norms=empty,
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
            )
            return _with_status((grads, grad_aux), grads_were_finite=True), state
        return _with_status(grads, grads_were_finite=True), state

    # Use PyTorch's grad_and_value (returns (grad, value) or (grad, (value, aux)))
    grad_and_value_fn = grad_and_value(loss_fn, argnums=argnums, has_aux=True)

    def grad_fn(*args, **kwargs):
        # Keep the graph only when an enclosing transform differentiates it.
        values_only = not under_differentiating_transform(when_compiling=True)
        with torch.no_grad() if values_only else contextlib.nullcontext():
            grad, value_and_aux = grad_and_value_fn(*args, **kwargs)
        result = pre_clipping_transform(grad)
        if return_aux or _force_grad_norms or return_pre_clipping_finite:
            # Return dict aux from per-example grad_fn; clipping-related norms are
            # produced by clipped_fun to avoid duplicating norm computation logic.
            # PyTorch vmap cannot handle namedtuples with None values when out_dims != None
            aux_dict = {}
            if return_pre_clipping_finite:
                aux_dict["_pre_clipping_finite"] = _pytree_finite_indicator(result)
            if return_aux:
                aux_dict["values"] = value_and_aux[0]
            if return_aux and has_aux:
                aux_dict["value_aux"] = value_and_aux[1]
            return result, aux_dict
        return result

    clipped_grad_fn, clip_state = clipped_fun(
        grad_fn,
        has_aux=return_aux or _force_grad_norms or return_pre_clipping_finite,
        batch_argnums=batch_argnums,
        clipping_norm=clipping_norm,
        normalize_by=normalize_by,
        return_aux=return_aux or return_pre_clipping_finite,
        second_moment=second_moment,
        microbatch_size=microbatch_size,
        dtype=dtype,
        compute_dtype=compute_dtype,
        _scale_fn=_scale_fn,
        _return_stats=_return_stats,
    )

    if return_pre_clipping_finite:

        def grad_fn_wrapper(*args, state, **kwargs):
            if batch_size_from_args(args, batch_argnums_tuple) == 0:
                return _empty_batch_response(args, state)
            with record_function("opaque::clipped_grad"):
                (clipped_grads, aux), returned_state = clipped_grad_fn(
                    *args, state=state, **kwargs
                )
            finite = aux._pre_clipping_finite
            if finite is None:
                raise RuntimeError(
                    "clipped_grad did not return the requested finiteness status."
                )
            status = ClippedGradStatus(grads_were_finite=bool(finite.all().item()))
            if return_aux:
                grad_aux = ClippedGradAux(
                    loss_values=aux.values,
                    grad_norms=aux.norms,
                    clipped_grad_norms=aux.clipped_norms,
                    loss_aux=aux.value_aux,
                    clipping_rate=aux.clipping_rate,
                    batch_size=aux.batch_size,
                    group_norms=aux.group_norms,
                )
                return (clipped_grads, grad_aux, status), returned_state
            return (clipped_grads, status), returned_state

        return grad_fn_wrapper, clip_state

    # clipped_grad_fn is now a callable, clip_state is a FixedClipState
    # Wrap the result to convert ClippedFunAux to ClippedGradAux
    if not return_aux:
        # No aux, return wrapped directly with state-passing signature
        def grad_fn_wrapper(*args, state, **kwargs):
            if batch_size_from_args(args, batch_argnums_tuple) == 0:
                return _empty_batch_response(args, state)

            with record_function("opaque::clipped_grad"):
                (result, returned_state) = clipped_grad_fn(*args, state=state, **kwargs)
            if _force_grad_norms:
                if isinstance(result, tuple):
                    stats = None
                    if _return_stats:
                        if len(result) == 3:
                            clipped_grads, aux, stats = result
                        else:
                            clipped_grads, stats = result
                            aux = None
                    else:
                        clipped_grads, aux = result
                    grad_aux = ClippedGradAux(
                        loss_values=None,
                        grad_norms=(aux.norms if aux is not None else None),
                        clipped_grad_norms=(
                            aux.clipped_norms if aux is not None else None
                        ),
                        loss_aux=None,
                        clipping_rate=(
                            aux.clipping_rate
                            if aux is not None
                            else (
                                stats.clipping_rate
                                if stats is not None
                                and isinstance(stats.clipping_rate, float)
                                else None
                            )
                        ),
                        batch_size=(
                            aux.batch_size
                            if aux is not None
                            else (stats.batch_size if stats is not None else 0)
                        ),
                        group_norms=(aux.group_norms if aux is not None else None),
                    )
                    if _return_stats:
                        return (clipped_grads, grad_aux, stats), returned_state
                    return (clipped_grads, grad_aux), returned_state
                return result, returned_state
            return result, returned_state

        return grad_fn_wrapper, clip_state
    else:
        # Need to convert ClippedFunAux to ClippedGradAux
        def grad_fn_wrapper(*args, state, **kwargs):
            if batch_size_from_args(args, batch_argnums_tuple) == 0:
                return _empty_batch_response(args, state)

            with record_function("opaque::clipped_grad"):
                (clipped_grads, aux), returned_state = clipped_grad_fn(
                    *args, state=state, **kwargs
                )
            grad_aux = ClippedGradAux(
                loss_values=aux.values,
                grad_norms=aux.norms,
                clipped_grad_norms=aux.clipped_norms,
                loss_aux=aux.value_aux,
                clipping_rate=aux.clipping_rate,
                batch_size=aux.batch_size,
                group_norms=aux.group_norms,
            )
            return (clipped_grads, grad_aux), returned_state

        return grad_fn_wrapper, clip_state


__all__ = ["ClippedGradAux", "ClippedGradStatus", "clipped_grad"]
