"""Per-example clipping and summing for arbitrary functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opaque.api.engine import autodiff, ops
from opaque.api.engine.clipping._helpers import normalize_to_tuple
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.pytree import global_norm, tree_leaves, tree_map
from opaque.api.engine.types import (
    ClippedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    clipped,
)
from opaque.api.engine.types import ClipState as _ClipState

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FixedClipState(_ClipState):
    """Marker state for fixed (non-adaptive) clipping.

    Returned by :func:`clipped_fun` and :func:`opaque.api.engine.clipping.clipped_grad`.
    Carries no fields; the configured clipping threshold flows through
    the ``ClippedPytree.max_norm`` metadata, not through the state.
    """


@dataclass(frozen=True)
class ClippedFunAux:
    """Diagnostic outputs from clipped_fun.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use the
    returned ``ClippedPytree.max_norm`` metadata for noise calibration.

    Fields:
        values: Per-example function values before clipping.
        norms: Per-example L2 norms before clipping.
        clipped_norms: Per-example L2 norms after clipping.
        value_aux: Per-example auxiliary payload returned by the wrapped function.
        clipping_rate: Fraction of per-example outputs whose norm exceeded the
            clipping threshold.  Equal to ``num_clipped / batch_size``.
        batch_size: Number of examples in the batch.
        group_norms: Per-group per-example L2 norms before clipping
            (dict[str, Tensor] with shape [batch_size] per group), or None
            when global clipping is used.
    """

    values: Any | None = None
    norms: Any | None = None
    clipped_norms: Any | None = None
    value_aux: Any | None = None
    clipping_rate: float | None = None
    batch_size: int = 0
    group_norms: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClippingStats:
    """Aggregated clipping statistics without per-example materialization."""

    num_clipped: float | dict[str, float]
    clipping_rate: float | dict[str, float] | None
    batch_size: int = 0
    all_finite: bool = True
    """Whether every pre-clipping output in the batch was finite."""


def _resolve_compute_dtype(
    tensor: Any,
    compute_dtype: Any | None,
) -> Any | None:
    """Resolve safe compute dtype for reductions.

    If compute_dtype is explicitly requested, use it. Otherwise, promote
    low-precision floating reductions (fp16/bf16) to float32 for numerical
    stability. Returns ``None`` to mean "no promotion needed" so the provider
    can use its default reduction behavior.
    """
    if compute_dtype is not None:
        return compute_dtype
    if ops.is_floating(tensor) and ops.is_low_precision(tensor):
        return ops.float32()
    return None


def _all_finite(value: Any) -> Any:
    """Return whether every floating-point array leaf is finite.

    The result stays an array so the check composes under the per-example
    transform; ``minimum`` on boolean arrays is the neutral spelling of a
    logical AND across leaves.
    """
    result: Any | None = None
    for leaf in tree_leaves(value):
        if not ops.is_array(leaf):
            continue
        finite = (
            ops.all(ops.isfinite(leaf))
            if ops.is_floating(leaf) or ops.is_complex(leaf)
            # Integer and boolean leaves cannot carry NaN or Inf.
            else ops.scalar(True, like=leaf)
        )
        result = finite if result is None else ops.minimum(result, finite)
    if result is None:
        raise TypeError("Expected at least one array leaf when checking finiteness.")
    return result


def _sum_clipped_tensor(
    tensor: Any,
    *,
    dim: int,
    output_dtype: Any | None,
    compute_dtype: Any | None,
) -> Any:
    """Sum with separate compute (accumulation) and output dtype.

    ``compute_dtype`` controls the reduction precision; ``output_dtype`` the
    caller-visible result dtype.  Defaults preserve the type-stable contract
    (output dtype = input dtype) with auto-fp32 promotion for bf16/fp16 inputs.
    """
    accum_dtype = _resolve_compute_dtype(tensor, compute_dtype)
    summed = ops.sum(tensor, axis=dim, dtype=accum_dtype)

    target = output_dtype if output_dtype is not None else ops.dtype(tensor)
    if ops.dtype(summed) != target:
        return ops.astype(summed, target)
    return summed


class _MicrobatchAccumulator:
    """Running sum over microbatches, held at the accumulation precision.

    The sum stays in the wider of ``compute_dtype`` and the output dtype and is
    cast to the output dtype once, at the end.
    """

    __slots__ = ("_compute_dtype", "_output_dtype", "_targets", "_total")

    def __init__(
        self,
        *,
        output_dtype: Any | None,
        compute_dtype: Any | None,
    ) -> None:
        self._output_dtype = output_dtype
        self._compute_dtype = compute_dtype
        self._total: Any | None = None
        self._targets: Any | None = None

    def _accum_dtype(self, x: Any) -> Any | None:
        """Never below the requested output precision, or the sum loses it."""
        resolved = _resolve_compute_dtype(x, self._compute_dtype)
        if self._output_dtype is None:
            return resolved
        if resolved is None:
            resolved = ops.dtype(x)
        return ops.promote_dtype(resolved, self._output_dtype)

    def add(self, values: Any) -> None:
        """Add one microbatch, summed over its batch dimension."""
        if self._targets is None:
            self._targets = tree_map(
                lambda x: (
                    ops.dtype(x) if self._output_dtype is None else self._output_dtype
                ),
                values,
            )
        partial = tree_map(
            lambda x: ops.sum(x, axis=0, dtype=self._accum_dtype(x)),
            values,
        )
        self._total = (
            partial
            if self._total is None
            else tree_map(lambda acc, new: ops.add(acc, new), self._total, partial)
        )

    def result(self) -> Any:
        """The accumulated sum in the caller-visible dtype, or None if unused."""
        if self._total is None:
            return None
        return tree_map(
            lambda acc, target: (
                acc if ops.dtype(acc) == target else ops.astype(acc, target)
            ),
            self._total,
            self._targets,
        )


def _validate_clipping_norm(clipping_norm: float | PerGroup) -> None:
    if isinstance(clipping_norm, PerGroup):
        for group_name, value in clipping_norm.values.items():
            if value <= 0:
                raise ValueError(
                    "clipping_norm must be positive for all groups, "
                    f"got {value} for group '{group_name}'"
                )
        return
    if clipping_norm <= 0:
        raise ValueError(f"clipping_norm must be positive, got {clipping_norm}")


def _microbatch_accumulate(
    per_example_fn,
    args,
    batch_argnums,
    in_dims,
    vmap_transform,
    microbatch_size,
    return_aux,
    dtype,
    compute_dtype,
    second_moment: bool = False,
):
    """Process batch in microbatches, accumulating results without materializing full batch.

    This implementation processes the batch in chunks of `microbatch_size`, accumulating
    results according to their type:
    - Clipped values: SUM (into a running accumulator in `compute_dtype`)
    - Auxiliary outputs: CONCAT (keep per-example for privacy analysis)

    Args:
        per_example_fn: Function to vmap over each example
        args: Full batch arguments
        batch_argnums: Which arguments contain batch dimension
        in_dims: Input dimensions for vmap
        vmap_transform: Vectorizing-transform constructor resolved at
            factory time (the provider's raw factory when a backend is
            active, else the deferred dispatcher)
        microbatch_size: Size of each microbatch
        return_aux: Whether function returns auxiliary outputs
        dtype: Optional output dtype for the accumulated pytree.  ``None``
            keeps the output in the input dtype (type-stable).
        compute_dtype: Optional internal accumulation dtype for the
            across-microbatch sum.  ``None`` (the default) auto-promotes
            bf16/fp16 inputs to float32 for numerical stability while
            still returning the result in the ``dtype`` (or input dtype).
            Independent of ``dtype``: ``compute_dtype=fp32`` with
            ``dtype=None`` accumulates in fp32 internally and casts back
            to input dtype at the boundary.
        second_moment: Whether to accumulate per-example gradient second
            moments alongside the clipped sum.

    Returns:
        Tuple of (accumulated_values, concatenated_aux)
    """
    # Get batch size from first batch argument
    from opaque.api.engine.clipping._helpers import batch_size_from_args

    batch_size = batch_size_from_args(args, batch_argnums)

    # Initialize accumulators
    grad_acc = _MicrobatchAccumulator(output_dtype=dtype, compute_dtype=compute_dtype)
    squared_acc = _MicrobatchAccumulator(
        output_dtype=dtype, compute_dtype=compute_dtype
    )
    aux_list = []

    # Process each microbatch
    for start_idx in range(0, batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, batch_size)

        # Slice batch arguments for this microbatch
        microbatch_args = list(args)
        for i in batch_argnums:
            microbatch_args[i] = tree_map(
                lambda x, s=start_idx, e=end_idx: (
                    ops.slice_array(x, slice(s, e)) if ops.is_array(x) else x
                ),
                args[i],
            )

        # vmap over microbatch.  Output shape depends on the orthogonal
        # ``second_moment`` and ``return_aux`` flags:
        #   (False, False) → clipped_values (which may itself be a pytree)
        #   (True,  False) → (clipped_values, squared_values)
        #   (False, True ) → (clipped_values, aux)
        #   (True,  True ) → (clipped_values, squared_values, aux)
        n_outputs = 1 + int(bool(second_moment)) + int(return_aux)
        out_dims = 0 if n_outputs == 1 else (0,) * n_outputs
        vmapped = vmap_transform(
            per_example_fn,
            in_axes=in_dims,
            out_axes=out_dims,
        )
        outputs = vmapped(*microbatch_args)
        if n_outputs == 1:
            clipped_values = outputs
            squared_values = None
            aux = ()
        else:
            idx = 0
            clipped_values = outputs[idx]
            idx += 1
            squared_values = outputs[idx] if second_moment else None
            if second_moment:
                idx += 1
            aux = outputs[idx] if return_aux else ()

        grad_acc.add(clipped_values)
        if second_moment:
            squared_acc.add(squared_values)

        # Collect aux outputs (CONCAT) - keep per-example
        if return_aux:
            aux_list.append(aux)

    # Concatenate aux across all microbatches
    if return_aux:
        # Concatenate aux outputs along batch dimension
        # Need to handle the list structure properly - transpose list of pytrees into pytree of lists
        def concat_leaves(*leaf_values):
            """Concatenate corresponding leaf values across microbatches."""
            if all(ops.is_array(v) for v in leaf_values):
                return ops.concatenate(leaf_values, axis=0)
            # Non-tensor leaves are assumed to be identical across microbatches;
            # return a single representative value to preserve the original structure.
            return leaf_values[0]

        aux = tree_map(concat_leaves, *aux_list)
    else:
        aux = ()

    return grad_acc.result(), squared_acc.result(), aux


def _microbatch_accumulate_stats_only(
    per_example_fn,
    args,
    batch_argnums,
    in_dims,
    vmap_transform,
    microbatch_size,
    dtype,
    compute_dtype,
    clipping_norm: float | PerGroup,
    second_moment: bool = False,
):
    """Process microbatches while accumulating only summed outputs and aggregate stats."""
    from opaque.api.engine.clipping._helpers import batch_size_from_args

    batch_size = batch_size_from_args(args, batch_argnums)

    grad_acc = _MicrobatchAccumulator(output_dtype=dtype, compute_dtype=compute_dtype)
    squared_acc = _MicrobatchAccumulator(
        output_dtype=dtype, compute_dtype=compute_dtype
    )
    total_batch_size = 0
    if isinstance(clipping_norm, PerGroup):
        total_num_clipped: float | dict[str, float] = dict.fromkeys(
            clipping_norm.values, 0.0
        )
    else:
        total_num_clipped = 0.0
    all_finite = True

    for start_idx in range(0, batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, batch_size)
        microbatch_args = list(args)
        for i in batch_argnums:
            microbatch_args[i] = tree_map(
                lambda x, s=start_idx, e=end_idx: (
                    ops.slice_array(x, slice(s, e)) if ops.is_array(x) else x
                ),
                args[i],
            )

        n_outputs = 2 + int(bool(second_moment))
        out_dims = (0,) * n_outputs
        vmapped = vmap_transform(
            per_example_fn,
            in_axes=in_dims,
            out_axes=out_dims,
        )
        outputs = vmapped(*microbatch_args)
        clipped_values = outputs[0]
        squared_values = outputs[1] if second_moment else None
        stats_aux = outputs[-1]

        grad_acc.add(clipped_values)
        if second_moment:
            squared_acc.add(squared_values)

        stats = _compute_clipping_stats(
            stats_aux["norms"],
            clipping_norm=clipping_norm,
            group_norms_dict=stats_aux.get("group_norms"),
            all_finite=stats_aux["all_finite"],
        )
        total_batch_size += stats.batch_size
        all_finite = all_finite and stats.all_finite
        if isinstance(total_num_clipped, dict):
            assert isinstance(stats.num_clipped, dict)
            for name, count in stats.num_clipped.items():
                total_num_clipped[name] += count
        else:
            assert isinstance(stats.num_clipped, float)
            total_num_clipped += stats.num_clipped

    if isinstance(total_num_clipped, dict):
        clipping_rate: float | dict[str, float]
        clipping_rate = {
            name: count / max(1.0, float(total_batch_size))
            for name, count in total_num_clipped.items()
        }
    else:
        clipping_rate = total_num_clipped / max(1.0, float(total_batch_size))

    return (
        grad_acc.result(),
        squared_acc.result(),
        ClippingStats(
            num_clipped=total_num_clipped,
            clipping_rate=clipping_rate,
            batch_size=total_batch_size,
            all_finite=all_finite,
        ),
    )


def _compute_clipping_stats(
    norms: Any | None,
    *,
    clipping_norm: float | PerGroup,
    group_norms_dict: dict[str, Any] | None,
    all_finite: Any | None = None,
) -> ClippingStats:
    """Compute aggregated clipping statistics from materialized norm tensors."""
    batch_size = ops.shape(norms)[0] if ops.is_array(norms) else 0
    finite = (
        bool(ops.scalar_item(ops.all(all_finite))) if all_finite is not None else True
    )
    if batch_size == 0:
        if isinstance(clipping_norm, PerGroup):
            empty_counts = dict.fromkeys(clipping_norm.values, 0.0)
            empty_rates = dict.fromkeys(clipping_norm.values, 0.0)
            return ClippingStats(
                num_clipped=empty_counts,
                clipping_rate=empty_rates,
                batch_size=0,
                all_finite=finite,
            )
        return ClippingStats(
            num_clipped=0.0,
            clipping_rate=0.0,
            batch_size=0,
            all_finite=finite,
        )

    if isinstance(clipping_norm, PerGroup) and group_norms_dict is not None:
        counts = {
            gname: float(
                ops.scalar_item(
                    ops.sum(
                        ops.astype(
                            ops.greater(gnorms, clipping_norm.values[gname]),
                            ops.float32(),
                        )
                    )
                )
            )
            for gname, gnorms in group_norms_dict.items()
        }
        rates = {gname: count / float(batch_size) for gname, count in counts.items()}
        return ClippingStats(
            num_clipped=counts,
            clipping_rate=rates,
            batch_size=batch_size,
            all_finite=finite,
        )

    effective_cn = (
        clipping_norm.effective
        if isinstance(clipping_norm, PerGroup)
        else clipping_norm
    )
    num_clipped = float(
        ops.scalar_item(
            ops.sum(ops.astype(ops.greater(norms, effective_cn), ops.float32()))
        )
    )
    return ClippingStats(
        num_clipped=num_clipped,
        clipping_rate=num_clipped / float(batch_size),
        batch_size=batch_size,
        all_finite=finite,
    )


def clipped_fun(
    fun: Callable[..., Any],
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    clipping_norm: float | PerGroup = 1.0,
    normalize_by: float = 1.0,
    return_aux: bool = False,
    return_stats: bool = False,
    second_moment: bool = False,
    microbatch_size: int | None = None,
    dtype: Any | None = None,
    compute_dtype: Any | None = None,
    _scale_fn: Callable | None = None,
) -> tuple[Callable, FixedClipState]:
    """Transform a function to clip its output and sum across a batch.

    This is the primary API for per-example clipping in DP-SGD. It wraps a function
    to clip each per-example output to a maximum L2 norm, then sums the clipped outputs.

    The returned pytree is wrapped as :class:`ClippedPytree` (single-stream)
    or :class:`SecondMomentClippingOutput` (paired-stream when
    ``second_moment=True``), carrying the post-normalization
    ``max_norm`` for downstream noise calibration.  The bound is part of
    the contract: consumers (``gaussian_noise``, ``mf_gaussian_noise``) read it
    directly without the caller threading a separate ``sensitivity``
    argument.  Unwrap to a raw pytree via ``.pytree`` if you need the
    summed values without metadata.

    Example Usage:
        After selecting a provider, pass a callable over native arrays to
        ``clipped_fun`` and invoke the returned function with native batched
        values and threaded state. Its ``ClippedPytree.pytree`` holds the
        aggregate, while ``ClippedPytree.max_norm`` holds the normalized bound.

    Formal Guarantees:
        For the first function output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be `clipping_norm`.
          Under replace-one DP, the sensitivity is doubled (2 * `clipping_norm`).
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
        clipping_norm: The maximum L2 norm allowed.
        normalize_by: Divide the clipped output by this value before returning.
        return_aux: If True, the returned Callable will return a per-example aux
            dataclass containing the original per-example values, per-example norms
            before clipping, and any auxiliary data returned by `fun`.
        return_stats: If True, return :class:`ClippingStats` with aggregate
            clipping statistics and whether every pre-clipping output was finite.
            Cannot be combined with ``return_aux``.
        second_moment: If True, also accumulate the element-wise sum of
            per-example squared clipped values, i.e. ``Σᵢ gᵢ²``.  The
            squaring happens inside the per-example loop so the
            second-stream sensitivity is the per-record squared bound
            ``C²`` (averaged: ``C² / normalize_by``).  The wrapped
            output becomes :class:`SecondMomentClippingOutput` with both
            streams.  Per-group ``clipping_norm`` is supported and yields
            ``SecondMomentClippingOutput`` with per-group ``max_norm``
            on both streams.
        microbatch_size: If set, the batch is split up into microbatches of this
            size for memory-efficient processing. Processes each microbatch separately
            and accumulates results without materializing the full batch of gradients.
            Set this to reduce peak memory usage at the cost of slightly slower computation.
            The running sum is held in ``compute_dtype``, so a bf16/fp16 run keeps
            one float32 copy of the summed output.  Pass a low-precision
            ``compute_dtype`` to give that memory back, at the cost of the
            accumulation precision.
        dtype: Optional dtype for the clipped+aggregated pytree. If None, the dtype
            will be the same as the dtypes of the function output.
        compute_dtype: Internal accumulation dtype for reductions (per-example
            clip-norm and the across-batch sum).  ``None`` (default) auto-promotes
            bf16/fp16 to float32 for numerical stability; explicit dtype forces
            that precision regardless of input.  Independent of ``dtype`` (which
            controls the *output* dtype).  Applies across microbatches too, so
            microbatched and non-microbatched runs agree to float32 precision.
    Returns:
        A tuple ``(clip_fn, FixedClipState)`` where ``clip_fn(*args, state=...)``
        clips the output of ``fun`` and sums across the batch.  The exact
        return shape depends on ``second_moment`` and ``return_aux``:

        | ``second_moment`` | ``return_aux`` | ``clip_fn`` returns                                |
        | :---------------- | :------------- | :------------------------------------------------- |
        | False             | False          | ``(ClippedPytree, state)``                         |
        | False             | True           | ``((ClippedPytree, ClippedFunAux), state)``        |
        | True              | False          | ``(SecondMomentClippingOutput, state)``            |
        | True              | True           | ``((SecondMomentClippingOutput, ClippedFunAux), state)`` |

        With ``return_stats=True`` and ``return_aux=False``, the clipped value
        is paired with :class:`ClippingStats`.
    """
    if return_aux and return_stats:
        raise ValueError("return_stats cannot be combined with return_aux=True")

    # Normalize batch_argnums to tuple
    batch_argnums = normalize_to_tuple(batch_argnums)

    # Wrap function to handle has_aux - use empty tuple () not None!
    if not has_aux:

        def fun_with_aux(*args, **kwargs):
            return (fun(*args, **kwargs), ())

    else:
        fun_with_aux = fun

    _validate_clipping_norm(clipping_norm)
    output_max_norm = clipping_norm / normalize_by
    output_squared_max_norm = (
        (clipping_norm * clipping_norm) / normalize_by if second_moment else None
    )
    clip_state = FixedClipState()

    # Resolve the provider's raw vmap constructor once at factory time so
    # per-call code (and compiled graphs) reach the framework transform
    # without going through dispatch.
    _vmap = autodiff.vmap_factory()

    def clipped_fn(*args, **kwargs):
        # Determine in_dims for vmap
        in_dims = tuple(0 if i in batch_argnums else None for i in range(len(args)))

        # Resolve scale function: default is fixed-norm clipping.
        # _scale_fn enables alternate bounding schemes (e.g. AUTO-S) while
        # reusing the vmap / microbatching / aux machinery below.
        scale_fn = (
            _scale_fn
            if _scale_fn is not None
            else (
                lambda v: clip_pytree(
                    v, clipping_norm=clipping_norm, compute_dtype=compute_dtype
                )
            )
        )

        # Define per-example function
        def per_example_fn(*args_single):
            value, aux = fun_with_aux(*args_single, **kwargs)
            clipped_value, norm = scale_fn(value)
            squared_value = (
                tree_map(
                    lambda x: ops.square(x) if ops.is_array(x) else x,
                    clipped_value,
                )
                if second_moment
                else None
            )
            if return_aux or return_stats:
                # Build aux dict with clipping metadata
                # IMPORTANT: Detach all tensors to prevent memory leaks from retaining
                # computational graphs. These are monitoring values, not used for gradients.
                aux_dict = {
                    "norms": ops.detach(norm.norm),
                    "clipped_norms": ops.detach(
                        global_norm(clipped_value, compute_dtype=compute_dtype)
                    ),
                }
                if return_stats:
                    aux_dict["all_finite"] = _all_finite(value)

                # Per-group norms (dict of scalar tensors → dict of 1D tensors after vmap)
                if norm.group_norms is not None:
                    aux_dict["group_norms"] = {
                        k: ops.detach(v) for k, v in norm.group_norms.items()
                    }

                # Extract nested values and aux from wrapped functions (e.g., grad_fn)
                # aux may be a dict like {"values": val, "value_aux": user_aux} or just user_aux
                if isinstance(aux, dict):
                    # Preserve "values" from nested dict if present (e.g., loss from grad_and_value)
                    if "values" in aux:
                        val = aux["values"]
                        aux_dict["values"] = (
                            ops.detach(val) if ops.is_array(val) else val
                        )
                    else:
                        # No nested "values", use function output
                        aux_dict["values"] = (
                            ops.detach(value) if ops.is_array(value) else value
                        )

                    # Extract user aux from nested dict if present
                    if has_aux:
                        if "value_aux" in aux:
                            aux_dict["value_aux"] = aux["value_aux"]
                        else:
                            # aux is already the user aux (not nested)
                            aux_dict["value_aux"] = aux
                else:
                    # aux is not a dict (direct user aux or None)
                    aux_dict["values"] = (
                        ops.detach(value) if ops.is_array(value) else value
                    )
                    if has_aux:
                        aux_dict["value_aux"] = aux

                if return_stats:
                    if second_moment:
                        return clipped_value, squared_value, aux_dict
                    return clipped_value, aux_dict

                if second_moment:
                    return clipped_value, squared_value, aux_dict
                return clipped_value, aux_dict
            if second_moment:
                return clipped_value, squared_value
            return clipped_value

        # Choose execution path based on microbatch_size
        stats = None
        if microbatch_size is None:
            # Fast path: vmap entire batch at once.  Output shape depends
            # on the (second_moment, return_aux) flags — see the per_example_fn
            # branches above.  When n_outputs == 1, vmap returns the single
            # pytree (which may itself be a tuple of tensors for tuple
            # params); when n_outputs > 1 the per_example_fn returns a
            # tuple of n_outputs pytrees.
            n_outputs = 1 + int(bool(second_moment)) + int(return_aux or return_stats)
            out_dims = 0 if n_outputs == 1 else (0,) * n_outputs
            vmapped = _vmap(
                per_example_fn,
                in_axes=in_dims,
                out_axes=out_dims,
            )
            outputs = vmapped(*args)
            if n_outputs == 1:
                clipped_values = outputs
                squared_values = None
                aux = ()
            else:
                idx = 0
                clipped_values = outputs[idx]
                idx += 1
                squared_values = outputs[idx] if second_moment else None
                if second_moment:
                    idx += 1
                aux = outputs[idx] if (return_aux or return_stats) else ()

            # Sum clipped values across batch dimension
            result = tree_map(
                lambda x: _sum_clipped_tensor(
                    x, dim=0, output_dtype=dtype, compute_dtype=compute_dtype
                ),
                clipped_values,
            )
            squared_result = (
                tree_map(
                    lambda x: _sum_clipped_tensor(
                        x, dim=0, output_dtype=dtype, compute_dtype=compute_dtype
                    ),
                    squared_values,
                )
                if second_moment
                else None
            )
        elif return_stats:
            result, squared_result, stats = _microbatch_accumulate_stats_only(
                per_example_fn=per_example_fn,
                args=args,
                batch_argnums=batch_argnums,
                in_dims=in_dims,
                vmap_transform=_vmap,
                microbatch_size=microbatch_size,
                dtype=dtype,
                compute_dtype=compute_dtype,
                clipping_norm=clipping_norm,
                second_moment=second_moment,
            )
            aux = ()
        else:
            # Manual microbatch accumulation: process in chunks, accumulate as we go
            result, squared_result, aux = _microbatch_accumulate(
                per_example_fn=per_example_fn,
                args=args,
                batch_argnums=batch_argnums,
                in_dims=in_dims,
                vmap_transform=_vmap,
                microbatch_size=microbatch_size,
                return_aux=return_aux,
                dtype=dtype,
                compute_dtype=compute_dtype,
                second_moment=second_moment,
            )

        # Normalize
        if normalize_by != 1.0:
            result = tree_map(lambda x: ops.divide(x, normalize_by), result)
            if second_moment:
                squared_result = tree_map(
                    lambda x: ops.divide(x, normalize_by), squared_result
                )

        if second_moment:
            output = SecondMomentClippingOutput(
                grads=ClippedPytree(pytree=result, max_norm=output_max_norm),
                squared_grads=ClippedPytree(
                    pytree=squared_result, max_norm=output_squared_max_norm
                ),
            )
        else:
            output = clipped(result, max_norm=output_max_norm)

        if not return_aux and not return_stats:
            return output

        if return_aux:
            aux_dict = aux if isinstance(aux, dict) else {}
            norms = aux_dict.get("norms")
            group_norms_dict = aux_dict.get("group_norms")
            stats = _compute_clipping_stats(
                norms, clipping_norm=clipping_norm, group_norms_dict=group_norms_dict
            )

            aux = ClippedFunAux(
                values=aux_dict.get("values"),
                norms=norms,
                clipped_norms=aux_dict.get("clipped_norms"),
                value_aux=aux_dict.get("value_aux"),
                clipping_rate=(
                    stats.clipping_rate
                    if isinstance(stats.clipping_rate, float)
                    or stats.clipping_rate is None
                    else None
                ),
                batch_size=stats.batch_size,
                group_norms=aux_dict.get("group_norms"),
            )
            return output, aux

        if microbatch_size is not None:
            assert stats is not None
            return output, stats

        aux_dict = aux if isinstance(aux, dict) else {}
        return output, _compute_clipping_stats(
            aux_dict.get("norms"),
            clipping_norm=clipping_norm,
            group_norms_dict=aux_dict.get("group_norms"),
            all_finite=aux_dict.get("all_finite"),
        )

    # Wrap function to accept and return state
    def stateful_clipped_fn(*args, state, **kwargs):
        result = clipped_fn(*args, **kwargs)
        return result, state  # State unchanged for fixed clipping

    # Return wrapped function with state
    return stateful_clipped_fn, clip_state


__all__ = ["ClippedFunAux", "ClippingStats", "clipped_fun"]
