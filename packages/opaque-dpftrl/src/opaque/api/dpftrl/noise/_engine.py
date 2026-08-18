"""Portable matrix-factorization noise execution for DP-FTRL."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opaque.api.engine import ops
from opaque.api.engine.backend import ensure_backend
from opaque.api.engine.pytree import (
    tree_flatten_with_paths,
    tree_map,
    tree_unflatten,
)
from opaque.random import fold_in as rng_fold_in
from opaque.random import normal
from opaque.types import NoiseState, PerGroup

from ._plan import MfExecutionPlan

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.random.types import RngKey


@dataclasses.dataclass(frozen=True)
class MFNoiseState(NoiseState):
    """Immutable state for provider-native matrix-factorization noise."""

    _inner_state: Any
    _step_counter: int
    _rng_key: RngKey
    _first_max_norm: float | PerGroup | None = None
    _first_max_norm_sync_fingerprint: int | None = None


def _internal_compute_dtype(dtype: object) -> object:
    """Promote low-precision native dtypes to the active provider's float32."""
    return ops.float32() if ops.is_low_precision(dtype) else dtype


def _resolve_compute_dtype(compute_dtype: object | None) -> object:
    return ops.float32() if compute_dtype is None else compute_dtype


def _validate_noise_stddev(stddev: float | PerGroup) -> None:
    if isinstance(stddev, PerGroup):
        for group_name, value in stddev.values.items():
            if value < 0:
                raise ValueError(
                    "MF noise standard deviation must be non-negative for all "
                    f"groups, got {value} for group {group_name!r}."
                )
        return
    if float(stddev) < 0:
        raise ValueError(
            f"MF noise standard deviation must be non-negative, got {stddev}."
        )


def _iid_normal_noise(
    target_tree: Any,
    stddev: float | PerGroup,
    *,
    key: RngKey,
    compute_dtype: object | None = None,
) -> Any:
    """Sample an IID Gaussian pytree with deterministic per-leaf keys."""
    _validate_noise_stddev(stddev)
    sample_dtype = _resolve_compute_dtype(compute_dtype)
    paths, leaves, treedef = tree_flatten_with_paths(target_tree)
    noise_leaves: list[Any] = []
    for leaf_index, (path, leaf) in enumerate(zip(paths, leaves, strict=True)):
        if not ops.is_array(leaf):
            raise TypeError(
                "MF noise expects native array leaves; "
                f"got {type(leaf).__name__} at path {path!r}."
            )
        leaf_stddev = stddev.for_path(path) if isinstance(stddev, PerGroup) else stddev
        if leaf_stddev == 0:
            noise = ops.zeros(ops.shape(leaf), dtype=sample_dtype, like=leaf)
        else:
            noise = normal(
                rng_fold_in(key, "mf_gaussian_leaf", leaf_index),
                ops.shape(leaf),
                dtype=sample_dtype,
                like=leaf,
            )
            noise = ops.multiply(noise, leaf_stddev)
        noise_leaves.append(noise)
    return tree_unflatten(treedef, noise_leaves)


def _tree_scale(tree: Any, scale: float) -> Any:
    if scale == 1.0:
        return tree
    return tree_map(lambda value: ops.multiply(value, scale), tree)


def _tree_add(left: Any, right: Any) -> Any:
    return tree_map(ops.add, left, right)


def _tree_subtract(left: Any, right: Any) -> Any:
    return tree_map(ops.subtract, left, right)


def _tree_linear_combination(
    trees: tuple[Any, ...], coefficients: tuple[float, ...]
) -> Any:
    if not trees or len(trees) != len(coefficients):
        raise ValueError(
            "linear-combination trees and coefficients must have equal non-zero length"
        )
    result = _tree_scale(trees[0], coefficients[0])
    for tree, coefficient in zip(trees[1:], coefficients[1:], strict=True):
        result = _tree_add(result, _tree_scale(tree, coefficient))
    return result


def _zero_tree_like(tree: Any, compute_dtype: object) -> Any:
    def zero(leaf: Any) -> Any:
        if not ops.is_array(leaf):
            raise TypeError(
                f"MF noise expects native array leaves; got {type(leaf).__name__}."
            )
        return ops.zeros(ops.shape(leaf), dtype=compute_dtype, like=leaf)

    return tree_map(zero, tree)


def _add_noise_tree(grads: Any, noise: Any, compute_dtype: object) -> Any:
    def add(grad: Any, noise_leaf: Any) -> Any:
        if not ops.is_array(grad):
            raise TypeError(
                f"MF noise expects native array leaves; got {type(grad).__name__}."
            )
        input_dtype = ops.dtype(grad)
        grad_compute = (
            ops.astype(grad, compute_dtype) if input_dtype != compute_dtype else grad
        )
        result = ops.add(grad_compute, noise_leaf)
        return (
            ops.astype(result, input_dtype) if input_dtype != compute_dtype else result
        )

    return tree_map(add, grads, noise)


def _initial_inner_state(
    plan: MfExecutionPlan, grad_template: Any, compute_dtype: object
) -> Any:
    if plan.mode == "toeplitz":
        return tuple(
            _zero_tree_like(grad_template, compute_dtype) for _ in range(plan.n_steps)
        )
    if plan.mode == "blt":
        return tuple(
            _zero_tree_like(grad_template, compute_dtype) for _ in plan.buffer_decay
        )
    return None


def _apply_plan(
    plan: MfExecutionPlan,
    iid_noise: Any,
    inner_state: Any,
    *,
    step: int,
    target_tree: Any,
    stddev: float | PerGroup,
    key: RngKey,
    compute_dtype: object,
) -> tuple[Any, Any]:
    if plan.mode == "identity":
        return iid_noise, None

    if plan.mode == "lambda_replay":
        if step == 0 or len(plan.inverse_coefficients) == 1:
            correlated = iid_noise
        else:
            previous = _iid_normal_noise(
                target_tree,
                stddev,
                key=rng_fold_in(key, "mf_gaussian_column", step - 1),
                compute_dtype=compute_dtype,
            )
            correlated = _tree_add(
                iid_noise,
                _tree_scale(previous, plan.inverse_coefficients[1]),
            )
        return _tree_scale(correlated, plan.column_scales[step]), None

    if plan.mode == "toeplitz":
        history = (*inner_state[:step], iid_noise, *inner_state[step + 1 :])
        count = min(step + 1, len(plan.inverse_coefficients))
        trees = tuple(reversed(history[: step + 1]))[:count]
        correlated = _tree_linear_combination(trees, plan.inverse_coefficients[:count])
        return _tree_scale(correlated, plan.column_scales[step]), history

    if plan.mode == "blt":
        buffers: tuple[Any, ...] = inner_state
        if buffers:
            buffered = _tree_linear_combination(buffers, plan.output_scale)
            correlated = _tree_subtract(iid_noise, buffered)
            next_buffers = tuple(
                _tree_add(_tree_scale(buffer, decay), correlated)
                for buffer, decay in zip(buffers, plan.buffer_decay, strict=True)
            )
        else:
            correlated = iid_noise
            next_buffers = ()
        return _tree_scale(correlated, plan.column_scales[step]), next_buffers

    raise ValueError(f"unsupported MF execution-plan mode: {plan.mode!r}")


def _check_mf_horizon(step: int, n_steps: int) -> None:
    """Raise if ``step`` is outside the calibrated MF horizon."""
    if step < 0 or step >= n_steps:
        raise ValueError(
            f"MF noise step {step} is outside the calibrated horizon "
            f"[0, {n_steps}). Rebuild the noise mechanism with a larger "
            f"n_steps, or stop calling noise_fn after {n_steps} iterations."
        )


def _require_positive_int_horizon(n_steps: object) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError(f"n_steps must be an int, got {type(n_steps).__name__}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return n_steps


def _matrix_factorization_noise(
    grad_template: Any,
    noising: MfExecutionPlan,
    *,
    key: RngKey,
    compute_dtype: object | None = None,
    n_steps: int | None = None,
) -> tuple[Callable[..., tuple[Any, MFNoiseState]], MFNoiseState]:
    """Create a portable raw MF noise function from an immutable host plan."""
    if not isinstance(noising, MfExecutionPlan):
        raise TypeError(
            "noising must be MfExecutionPlan; strategy runtime hooks and native "
            "matrix objects are not portable execution inputs"
        )
    horizon = noising.n_steps
    if n_steps is not None and _require_positive_int_horizon(n_steps) != horizon:
        raise ValueError(
            f"n_steps ({n_steps}) does not match execution-plan horizon ({horizon})."
        )
    ensure_backend(grad_template)
    resolved_compute_dtype = _resolve_compute_dtype(compute_dtype)
    state = MFNoiseState(
        _inner_state=_initial_inner_state(
            noising, grad_template, resolved_compute_dtype
        ),
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
        *,
        stddev: float | PerGroup,
    ) -> tuple[Any, MFNoiseState]:
        step = st._step_counter
        _check_mf_horizon(step, horizon)
        ensure_backend(clipped_grads)
        step_key = rng_fold_in(st._rng_key, "mf_gaussian_column", step)
        iid_noise = _iid_normal_noise(
            clipped_grads,
            stddev,
            key=step_key,
            compute_dtype=resolved_compute_dtype,
        )
        correlated, next_inner = _apply_plan(
            noising,
            iid_noise,
            st._inner_state,
            step=step,
            target_tree=clipped_grads,
            stddev=stddev,
            key=st._rng_key,
            compute_dtype=resolved_compute_dtype,
        )
        noisy_grads = _add_noise_tree(clipped_grads, correlated, resolved_compute_dtype)
        return noisy_grads, MFNoiseState(
            _inner_state=next_inner,
            _step_counter=step + 1,
            _rng_key=st._rng_key,
            _first_max_norm=st._first_max_norm,
            _first_max_norm_sync_fingerprint=st._first_max_norm_sync_fingerprint,
        )

    return noise_fn, state


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def _expect_clipped(value: Any, *, op: str):
    """Reject already-noised or non-clipped inputs."""
    from opaque.types import ClippedPytree, NoisedPytree

    if isinstance(value, NoisedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs, not NoisedPytree values that "
            "have already passed through a noise mechanism."
        )
    if not isinstance(value, ClippedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs. Wrap manual values with "
            "opaque.types.clipped(...)."
        )
    return value


def _validate_constant_max_norm(
    grads: Any,
    first_max_norm: float | PerGroup | None,
    *,
    op: str,
) -> float | PerGroup:
    """Latch the clipping bound and reject varying MF sensitivity."""
    max_norm = grads.max_norm
    if isinstance(max_norm, PerGroup):
        for group_name, value in max_norm.values.items():
            if value < 0:
                raise ValueError(
                    "ClippedPytree max_norm must be non-negative for all groups, "
                    f"got {value} for group {group_name!r}."
                )
    elif float(max_norm) < 0:
        raise ValueError(
            f"ClippedPytree max_norm must be non-negative, got {grads.max_norm}"
        )
    if first_max_norm is not None and max_norm != first_max_norm:
        raise ValueError(
            f"{op} saw a varying ClippedPytree.max_norm across calls "
            f"(first={first_max_norm}, now={max_norm}). MF privacy proofs "
            "assume a constant per-step sensitivity; this is satisfied by "
            "fixed and AUTO-S clipping but not by adaptive clipping."
        )
    return max_norm


__all__ = ["MFNoiseState"]
