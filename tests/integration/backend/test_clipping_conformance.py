"""Shared black-box conformance tests for engine clipping factories."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tests.integration.backend._harness import ClippingRun, run_clipping
from tests.integration.backend._providers import ProviderCase, provider_case

from opaque import pytree
from opaque.api.engine.clipping import per_group
from opaque.backend import use_backend
from opaque.precision import all_finite, loss_scaler
from opaque.types import (
    ClippedPytree,
    ClipState,
    PerGroup,
    SecondMomentClippingOutput,
    clipped,
)

_PROVIDERS = ("torch", "jax", "mlx")
_KINDS = ("fixed", "auto")
_BOUND = 0.75
_GAMMA = 0.05
_X_VALUES = np.array(
    [[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0], [3.0, -0.25], [1.5, 0.5]],
    dtype=np.float32,
)
_Y_VALUES = np.array([1.0, 0.0, 1.0, 0.0, -0.5], dtype=np.float32)


def _inputs(
    case: ProviderCase,
    *,
    dtype_name: str = "float32",
    x_values: Any | None = None,
    y_values: Any | None = None,
) -> tuple[Any, Any, Any]:
    dtype = case.dtype(dtype_name)
    params = {
        "layer": {"weight": case.array([0.25, -0.5], dtype)},
        "bias": case.array(0.1, dtype),
    }
    x = case.array(
        x_values if x_values is not None else _X_VALUES,
        dtype,
    )
    y = case.array(
        y_values if y_values is not None else _Y_VALUES,
        dtype,
    )
    return params, x, y


def _loss(params: Any, x: Any, y: Any) -> Any:
    prediction = x @ params["layer"]["weight"] + params["bias"]
    return (prediction - y) ** 2


def _loss_with_aux(params: Any, x: Any, y: Any) -> Any:
    prediction = x @ params["layer"]["weight"] + params["bias"]
    return (prediction - y) ** 2, {"prediction": prediction}


def _expected_scalar_bound_sum(kind: str) -> tuple[np.ndarray, np.ndarray]:
    summed = np.zeros(3, dtype=np.float64)
    for x, y in zip(_X_VALUES, _Y_VALUES, strict=True):
        error = x @ np.array([0.25, -0.5]) + 0.1 - y
        grad = np.concatenate((2.0 * error * x, np.array([2.0 * error])))
        norm = np.linalg.norm(grad)
        scale = min(1.0, _BOUND / norm) if kind == "fixed" else _BOUND / (norm + _GAMMA)
        summed += scale * grad
    return summed[:2].astype(np.float32), summed[2:].astype(np.float32).reshape(())


def _tree_leaves(case: ProviderCase, tree: Any) -> list[Any]:
    with use_backend(case.backend):
        leaves, _ = pytree.tree_flatten(tree)
    return leaves


def _assert_tree_close(
    case: ProviderCase,
    actual: Any,
    expected: Any,
    *,
    rtol: float = 2e-5,
    atol: float = 2e-5,
) -> None:
    with use_backend(case.backend):
        actual_paths, actual_leaves, _ = pytree.tree_flatten_with_paths(actual)
        expected_paths, expected_leaves, _ = pytree.tree_flatten_with_paths(expected)
    assert actual_paths == expected_paths
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(
            case.to_numpy(actual_leaf),
            case.to_numpy(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


def _assert_public_run(run: ClippingRun, *, max_norm: Any) -> ClippedPytree:
    assert isinstance(run.grads, ClippedPytree)
    assert isinstance(run.initial_state, ClipState)
    assert isinstance(run.state, ClipState)
    assert run.state == run.initial_state
    assert run.grads.max_norm == max_norm
    return run.grads


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_public_factories_support_nested_native_parameter_trees(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)

    run = run_clipping(
        case,
        _loss,
        params,
        x,
        y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )

    grads = _assert_public_run(run, max_norm=_BOUND)
    assert set(grads.pytree) == {"layer", "bias"}
    assert grads.pytree["layer"]["weight"].shape == (2,)
    assert grads.pytree["bias"].shape == ()
    assert all(
        leaf.dtype == case.dtype("float32") for leaf in _tree_leaves(case, grads.pytree)
    )
    expected_weight, expected_bias = _expected_scalar_bound_sum(kind)
    np.testing.assert_allclose(
        case.to_numpy(grads.pytree["layer"]["weight"]),
        expected_weight,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        case.to_numpy(grads.pytree["bias"]),
        expected_bias,
        rtol=2e-5,
        atol=2e-5,
    )
    with use_backend(case.backend):
        assert all_finite(grads)


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_public_factories_support_nested_per_group_bounds(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)
    with use_backend(case.backend):
        bounds = per_group(params, weight=0.5, bias=0.25)

    run = run_clipping(
        case,
        _loss,
        params,
        x,
        y,
        kind=kind,
        bound=bounds,
        gamma=_GAMMA,
        return_aux=True,
    )

    grads = _assert_public_run(run, max_norm=bounds)
    assert isinstance(grads.max_norm, PerGroup)
    assert run.aux.group_norms is not None
    assert set(run.aux.group_norms) == {"weight", "bias"}
    assert all(value.shape == (5,) for value in run.aux.group_norms.values())


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_uneven_microbatch_matches_full_batch(provider_name: str, kind: str) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)
    common = {
        "kind": kind,
        "bound": _BOUND,
        "gamma": _GAMMA,
        "return_aux": True,
    }

    full = run_clipping(case, _loss, params, x, y, **common)
    microbatched = run_clipping(case, _loss, params, x, y, microbatch_size=2, **common)

    _assert_tree_close(case, microbatched.grads.pytree, full.grads.pytree)
    np.testing.assert_allclose(
        case.to_numpy(microbatched.aux.grad_norms),
        case.to_numpy(full.aux.grad_norms),
        rtol=2e-5,
        atol=2e-5,
    )
    assert microbatched.aux.batch_size == full.aux.batch_size == 5


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_empty_batch_returns_shaped_zeros_and_stable_state(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, _, _ = _inputs(case)
    empty_x = case.array(np.empty((0, 2), dtype=np.float32), case.dtype("float32"))
    empty_y = case.array(np.empty((0,), dtype=np.float32), case.dtype("float32"))

    run = run_clipping(
        case,
        _loss_with_aux,
        params,
        empty_x,
        empty_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )

    grads = _assert_public_run(run, max_norm=_BOUND)
    for leaf in _tree_leaves(case, grads.pytree):
        np.testing.assert_array_equal(case.to_numpy(leaf), np.zeros(leaf.shape))
    assert run.aux.batch_size == 0
    assert run.aux.loss_values.shape == (0,)
    assert run.aux.grad_norms.shape == (0,)
    assert run.aux.clipped_grad_norms.shape == (0,)
    assert run.aux.loss_aux is None


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_auxiliary_output_is_batched_and_norms_are_bounded(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)

    run = run_clipping(
        case,
        _loss_with_aux,
        params,
        x,
        y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )

    assert run.aux.batch_size == 5
    assert run.aux.loss_values.shape == (5,)
    assert run.aux.grad_norms.shape == (5,)
    assert run.aux.clipped_grad_norms.shape == (5,)
    assert run.aux.loss_aux["prediction"].shape == (5,)
    clipped_norms = case.to_numpy(run.aux.clipped_grad_norms)
    assert np.isfinite(clipped_norms).all()
    assert (clipped_norms <= _BOUND + 2e-5).all()


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_second_moment_preserves_public_wrappers(provider_name: str, kind: str) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)

    run = run_clipping(
        case,
        _loss,
        params,
        x,
        y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        normalize_by=5.0,
        second_moment=True,
    )

    assert isinstance(run.grads, SecondMomentClippingOutput)
    assert run.grads.grads.max_norm == pytest.approx(_BOUND / 5.0)
    assert run.grads.squared_grads.max_norm == pytest.approx(_BOUND**2 / 5.0)
    assert run.grads.grads.pytree["layer"]["weight"].shape == (2,)
    assert run.grads.squared_grads.pytree["layer"]["weight"].shape == (2,)
    for leaf in _tree_leaves(case, run.grads.squared_grads.pytree):
        assert (case.to_numpy(leaf) >= 0).all()
    with use_backend(case.backend):
        assert all_finite(run.grads)


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_multiple_batch_arguments_and_explicit_argnums(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, x, y = _inputs(case)
    weight = params["layer"]["weight"]
    bias = params["bias"]

    def loss(weight: Any, bias: Any, x: Any, y: Any) -> Any:
        return (x @ weight + bias - y) ** 2

    run = run_clipping(
        case,
        loss,
        weight,
        bias,
        x,
        y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        argnums=(0, 1),
        batch_argnums=(2, 3),
    )

    grads = _assert_public_run(run, max_norm=_BOUND)
    assert isinstance(grads.pytree, tuple)
    assert grads.pytree[0].shape == weight.shape
    assert grads.pytree[1].shape == bias.shape


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_nonfinite_per_example_gradients_are_sanitized(
    provider_name: str, kind: str
) -> None:
    case = provider_case(provider_name)
    params, _, _ = _inputs(case)
    x = case.array(
        [[1.0, -1.0], [float("nan"), 0.5], [float("inf"), -2.0]],
        case.dtype("float32"),
    )
    y = case.array([0.0, 1.0, 0.0], case.dtype("float32"))

    run = run_clipping(
        case,
        _loss,
        params,
        x,
        y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        return_aux=True,
    )

    with use_backend(case.backend):
        assert all_finite(run.grads)
    assert np.isfinite(case.to_numpy(run.aux.clipped_grad_norms)).all()


@pytest.mark.parametrize("provider_name", _PROVIDERS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_low_precision_defaults_preserve_dtype_and_accumulate_to_fp32(
    provider_name: str, kind: str, dtype_name: str
) -> None:
    case = provider_case(provider_name)
    x_values = np.tile([[0.1, -0.2], [0.3, 0.05], [-0.15, 0.4]], (33, 1))
    y_values = np.tile([0.2, -0.1, 0.3], 33)
    params_low, x_low, y_low = _inputs(
        case,
        dtype_name=dtype_name,
        x_values=x_values,
        y_values=y_values,
    )
    params_fp32, x_fp32, y_fp32 = _inputs(
        case,
        x_values=x_values,
        y_values=y_values,
    )

    default = run_clipping(
        case,
        _loss,
        params_low,
        x_low,
        y_low,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )
    promoted = run_clipping(
        case,
        _loss,
        params_low,
        x_low,
        y_low,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        dtype=case.dtype("float32"),
    )
    reference = run_clipping(
        case,
        _loss,
        params_fp32,
        x_fp32,
        y_fp32,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        dtype=case.dtype("float32"),
    )

    assert all(
        leaf.dtype == case.dtype(dtype_name)
        for leaf in _tree_leaves(case, default.grads.pytree)
    )
    assert all(
        leaf.dtype == case.dtype("float32")
        for leaf in _tree_leaves(case, promoted.grads.pytree)
    )
    tolerance = 2e-2 if dtype_name == "float16" else 1.5e-1
    _assert_tree_close(
        case,
        promoted.grads.pytree,
        reference.grads.pytree,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_loss_scaler_and_all_finite_are_backend_neutral(provider_name: str) -> None:
    case = provider_case(provider_name)
    scaler, state = loss_scaler(
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
    )
    grads = {
        "nested": [case.array([8.0, -16.0], case.dtype("float32"))],
        "step": case.array(3, case.dtype("int64")),
    }

    with use_backend(case.backend):
        scaled_loss = scaler.scale_loss(case.array(0.5, case.dtype("float32")), state)
        unscaled = scaler.unscale_grads(grads, state)
        assert all_finite(unscaled)
        assert not all_finite(
            clipped(
                {"w": case.array([1.0, float("nan")], case.dtype("float32"))},
                max_norm=1.0,
            )
        )
        assert not all_finite(
            {"w": case.array([1.0, float("inf")], case.dtype("float32"))}
        )
        finite_stream = clipped(
            {"w": case.array([1.0], case.dtype("float32"))}, max_norm=1.0
        )
        nonfinite_stream = clipped(
            {"w": case.array([float("nan")], case.dtype("float32"))}, max_norm=1.0
        )
        assert not all_finite(
            {
                "streams": SecondMomentClippingOutput(
                    grads=finite_stream,
                    squared_grads=nonfinite_stream,
                )
            }
        )

    assert case.to_numpy(scaled_loss).item() == pytest.approx(4.0)
    np.testing.assert_allclose(
        case.to_numpy(unscaled["nested"][0]), [1.0, -2.0], rtol=0, atol=0
    )
    assert unscaled["step"].dtype == grads["step"].dtype

    clean_once = scaler.update(state, grads_were_finite=True)
    grown = scaler.update(clean_once, grads_were_finite=True)
    backed_off = scaler.update(grown, grads_were_finite=False)
    assert state.scale == 8.0
    assert state.growth_tracker == 0
    assert clean_once.scale == 8.0
    assert clean_once.growth_tracker == 1
    assert grown.scale == 16.0
    assert grown.growth_tracker == 0
    assert backed_off.scale == 8.0
    assert backed_off.growth_tracker == 0
