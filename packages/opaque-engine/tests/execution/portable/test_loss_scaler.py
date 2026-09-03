"""Portable loss-scaler behavior for functional gradient PyTrees."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.clipping import clipped_grad
from opaque.precision import LossScaler, LossScalerState, all_finite, loss_scaler
from opaque.types import NoisedPytree, SecondMomentClippingOutput, clipped


def _grads(backend_case, magnitude: float = 1.0) -> dict[str, object]:
    return {
        "fp32": backend_case.array(
            [[magnitude, magnitude]], dtype=backend_case.dtype("float32")
        ),
        "bf16": backend_case.array([magnitude], dtype=backend_case.dtype("bfloat16")),
        "step": backend_case.array(7, dtype=backend_case.dtype("int64")),
    }


def _clipped_sum(params, samples, scaler: LossScaler, state: LossScalerState):
    def loss_fn(parameters, sample):
        return scaler.scale_loss(ops.sum(ops.multiply(parameters[0], sample)), state)

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
        pre_clipping_transform=lambda grads: scaler.unscale_grads(grads, state),
    )
    grads, _ = grad_fn(params, samples, state=clip_state)
    return grads.pytree[0]


def test_factory_validates_configuration_and_returns_immutable_state(
    backend_case,
) -> None:
    scaler, state = loss_scaler()
    assert isinstance(scaler, LossScaler)
    assert isinstance(state, LossScalerState)
    assert state == LossScalerState(scale=2**16, growth_tracker=0)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="init_scale"):
            loss_scaler(init_scale=bad)
    with pytest.raises(ValueError, match="growth_factor"):
        loss_scaler(growth_factor=1.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        loss_scaler(backoff_factor=1.0)
    with pytest.raises(ValueError, match="growth_interval"):
        loss_scaler(growth_interval=0)


def test_scale_and_unscale_preserve_dtypes_structure_and_complex_values(
    backend_case,
) -> None:
    scaler, state = loss_scaler(init_scale=128.0)
    loss = backend_case.array(0.5, dtype=backend_case.dtype("float32"))
    grads = _grads(backend_case, magnitude=128.0)
    grads["complex"] = backend_case.array(
        [complex(128.0, 128.0)], dtype=backend_case.dtype("complex64")
    )

    scaled_loss = scaler.scale_loss(loss, state)
    unscaled = scaler.unscale_grads(grads, state)

    backend_case.assert_allclose(scaled_loss, 64.0)
    assert set(unscaled) == set(grads)
    assert unscaled["fp32"].dtype == backend_case.dtype("float32")
    assert unscaled["bf16"].dtype == backend_case.dtype("bfloat16")
    assert unscaled["step"] is grads["step"]
    backend_case.assert_allclose(unscaled["fp32"], [[1.0, 1.0]])
    backend_case.assert_allclose(unscaled["complex"], [complex(1.0, 1.0)])


def test_unscale_before_clipping_preserves_the_dp_gradient_bound(backend_case) -> None:
    params = (backend_case.array(1.0, dtype=backend_case.dtype("float32")),)
    samples = backend_case.array([1.0, 2.0, 3.0], dtype=backend_case.dtype("float32"))
    baseline_scaler, baseline_state = loss_scaler(enabled=False)
    scaled_scaler, scaled_state = loss_scaler(init_scale=128.0)

    baseline = _clipped_sum(params, samples, baseline_scaler, baseline_state)
    scaled = _clipped_sum(params, samples, scaled_scaler, scaled_state)

    backend_case.assert_allclose(scaled, backend_case.to_host(baseline))


def test_all_finite_covers_arrays_wrappers_and_integer_passthrough(
    backend_case,
) -> None:
    finite = _grads(backend_case)
    assert all_finite(finite) is True
    assert all_finite({"value": backend_case.array([float("inf")])}) is False
    assert all_finite({"value": backend_case.array([float("nan")])}) is False
    assert (
        all_finite(
            {
                "value": backend_case.array(
                    [complex(float("inf"), 0.0)], dtype=backend_case.dtype("complex64")
                )
            }
        )
        is False
    )
    assert all_finite(
        {"step": backend_case.array(2**60, dtype=backend_case.dtype("int64"))}
    )
    assert (
        all_finite(clipped({"w": backend_case.array([float("nan")])}, max_norm=1.0))
        is False
    )
    assert (
        all_finite(
            NoisedPytree(
                pytree={"w": backend_case.array([float("inf")])},
                max_norm=1.0,
                noise_stddev=0.5,
            )
        )
        is False
    )
    assert (
        all_finite(
            SecondMomentClippingOutput(
                grads=clipped({"w": backend_case.array([1.0])}, max_norm=1.0),
                squared_grads=clipped(
                    {"w": backend_case.array([float("nan")])}, max_norm=1.0
                ),
            )
        )
        is False
    )


def test_update_growth_backoff_and_disabled_mode_are_pure(backend_case) -> None:
    scaler, state = loss_scaler(init_scale=128.0, growth_factor=2.0, growth_interval=3)
    first = scaler.update(state, grads_were_finite=True)
    second = scaler.update(first, grads_were_finite=True)
    grown = scaler.update(second, grads_were_finite=True)
    backed_off = scaler.update(grown, grads_were_finite=False)

    assert state == LossScalerState(scale=128.0, growth_tracker=0)
    assert first == LossScalerState(scale=128.0, growth_tracker=1)
    assert second == LossScalerState(scale=128.0, growth_tracker=2)
    assert grown == LossScalerState(scale=256.0, growth_tracker=0)
    assert backed_off == LossScalerState(scale=128.0, growth_tracker=0)

    disabled, disabled_state = loss_scaler(enabled=False, init_scale=128.0)
    loss = backend_case.array(0.5, dtype=backend_case.dtype("float32"))
    grads = _grads(backend_case)
    assert disabled.scale_loss(loss, disabled_state) is loss
    assert disabled.unscale_grads(grads, disabled_state) is grads
    assert disabled.update(disabled_state, grads_were_finite=False) is disabled_state
